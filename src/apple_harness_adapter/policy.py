"""Text-only Inkling 2 Mini policy backed by MLX-LM."""

import json
from math import isfinite
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.inkling2 import layers
from mlx_lm.tuner.utils import linear_to_lora_layers

from .trace import Turn


class InklingPolicy:
    prefill_step_size = 2048

    def __init__(self, model_path: str, revision: str | None, rank: int, seed: int):
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        mx.random.seed(seed)
        self.model, self.tokenizer = load(model_path, revision=revision)
        language_model = self.model.model.language_model
        if self.model.model_type != "inkling2_loop" or not language_model.args.loop_config:
            raise ValueError("This backend expects a looped Inkling 2 Mini checkpoint")

        # The current fused Inkling kernels are for inference. Use the MLX reference
        # operations for both sampling and training so both paths use the same router.
        layers._use_kernels = lambda: False
        layers.TopKRouter.__call__ = differentiable_router
        self.model.freeze()
        core = language_model.args.loop_core
        self.target = f"model.language_model.layers.{core.stop - 1}.self_attn.o_proj"
        self.adapter_config = {
            "fine_tune_type": "lora",
            "num_layers": 0,
            "lora_parameters": {"rank": rank, "scale": 1.0, "dropout": 0.0, "keys": [self.target]},
        }
        linear_to_lora_layers(self.model, 0, self.adapter_config["lora_parameters"])
        self.model.eval()
        parameters = dict(tree_flatten(self.model.trainable_parameters()))
        if set(parameters) != {f"{self.target}.lora_a", f"{self.target}.lora_b"}:
            raise RuntimeError("Only the two selected LoRA matrices should be trainable")
        self.trainable_count = sum(value.size for value in parameters.values())

    def prompt_ids(self, messages: list[dict], tools: list[dict] | None, effort: str) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=effort,
        )
        return self.tokenizer.encode(rendered, add_special_tokens=False)

    def sample(self, prompt: list[int], limit: int, temperature: float) -> Turn:
        if not prompt or limit < 1 or not isfinite(temperature) or temperature <= 0:
            raise ValueError(
                "Sampling needs a prompt, positive token limit and positive temperature"
            )

        def sampler(logprobs):
            return mx.random.categorical(logprobs / temperature)

        tokens, logprobs = [], []
        stream = generate_step(
            mx.array(prompt),
            self.model,
            max_tokens=limit,
            sampler=sampler,
            stream=mx.default_stream(mx.default_device()),
            prefill_step_size=self.prefill_step_size,
        )
        for token, raw_logprobs in stream:
            scaled = raw_logprobs / temperature
            logprobs.append((scaled[token] - mx.logsumexp(scaled)).item())
            tokens.append(token)
            if token in self.tokenizer.eos_token_ids:
                break
        return Turn(tuple(prompt), tuple(tokens), tuple(logprobs), temperature)

    def decode(self, tokens: tuple[int, ...]) -> str:
        return self.tokenizer.decode(list(tokens), skip_special_tokens=False)

    @staticmethod
    def score(model: nn.Module, turn: Turn) -> mx.array:
        # Shift by one so the final prompt token predicts the first sampled token.
        # Each request is scored separately because a later harness prompt can rewrite history.
        start = len(turn.prompt_token_ids) - 1
        inputs = mx.array(turn.input_ids[:-1])[None, :]
        logits = model(inputs)[0, start:, :].astype(mx.float32) / turn.temperature
        targets = mx.array(turn.completion_token_ids)[:, None]
        selected = mx.take_along_axis(logits, targets, axis=-1)[:, 0]
        return selected - mx.logsumexp(logits, axis=-1)

    def replay_sampling_logprobs(self, turn: Turn) -> mx.array:
        # Verify the sampler's probabilities with its original prefill and
        # one-token decode shapes, outside gradient tracking.
        cache = self.model.make_cache()
        start = len(turn.prompt_token_ids) - 1
        inputs = mx.array(turn.input_ids[:-1])[None, :]
        for offset in range(0, start, self.prefill_step_size):
            end = min(offset + self.prefill_step_size, start)
            self.model(inputs[:, offset:end], cache=cache)
            mx.eval([visit.state for visit in cache])

        selected = []
        for offset, token in enumerate(turn.completion_token_ids, start):
            logits = self.model(inputs[:, offset : offset + 1], cache=cache)[:, -1, :]
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            mx.eval(logprobs)
            scaled = logprobs.squeeze(0) / turn.temperature
            selected.append(scaled[token] - mx.logsumexp(scaled))
        return mx.stack(selected)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        weights = dict(tree_flatten(self.model.trainable_parameters()))
        mx.save_safetensors(str(directory / "adapters.safetensors"), weights)
        (directory / "adapter_config.json").write_text(
            json.dumps(self.adapter_config, indent=2) + "\n", encoding="utf-8"
        )

    def load_adapter(self, directory: Path) -> None:
        config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
        if config != self.adapter_config:
            raise ValueError(
                "Adapter configuration does not match the selected model and LoRA rank"
            )
        weights = mx.load(str(directory / "adapters.safetensors"))
        parameters = dict(tree_flatten(self.model.trainable_parameters()))
        if weights.keys() != parameters.keys() or any(
            value.shape != parameters[name].shape or value.dtype != parameters[name].dtype
            for name, value in weights.items()
        ):
            raise ValueError("Adapter weights do not match the selected LoRA matrices")
        self.model.load_weights(list(weights.items()), strict=False)
        mx.eval(self.model.trainable_parameters())


def differentiable_router(router, hidden_states):
    scores = mx.sigmoid(hidden_states.astype(mx.float32) @ router.weight.T)
    # Expert selection is discrete. MLX can differentiate the gathered weights only
    # when the indices are detached from the graph.
    ranking = mx.stop_gradient(scores + router.e_score_correction_bias)
    indices = mx.argpartition(-ranking, kth=router.top_k - 1, axis=-1)[..., : router.top_k]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    return indices, router.routed_scaling_factor * (
        weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    )

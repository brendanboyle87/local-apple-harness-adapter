"""Shared MLX sampling, scoring, and adapter storage for local text policies."""

import json
from math import isfinite
from pathlib import Path
from time import monotonic

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import DEFAULT_ALLOW_PATTERNS

from ..telemetry import MODEL_REQUEST
from ..trace import Turn


class MLXPolicy:
    prefill_step_size = 2048
    default_rank = 8
    lora_scale = 1.0
    checkpoint_layers = None
    cached_gradient = None
    allow_legacy_adapter = False

    def __init__(self, model_path: str, revision: str | None, rank: int, seed: int):
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        mx.random.seed(seed)
        location = Path(model_path).expanduser()
        if location.is_dir():
            location = location.resolve()
            model_path = str(location)
        else:
            location = Path(
                snapshot_download(
                    model_path, revision=revision, allow_patterns=DEFAULT_ALLOW_PATTERNS
                )
            )
            revision = location.name
        self.model_name = self.serving_name or Path(model_path).name
        self.model_identity = {"model": model_path, "revision": revision}
        self.model, self.tokenizer = self.load_checkpoint(location)
        self.targets = tuple(self.configure_model())
        self.adapter_metadata = {
            "backend": self.backend,
            **self.model_identity,
            "lora_targets": list(self.targets),
        }
        self.model.freeze()
        self.adapter_config = {
            "fine_tune_type": "lora",
            "num_layers": 0,
            "lora_parameters": {
                "rank": rank,
                "scale": self.lora_scale,
                "dropout": 0.0,
                "keys": list(self.targets),
            },
        }
        linear_to_lora_layers(self.model, 0, self.adapter_config["lora_parameters"])
        self.model.eval()
        parameters = dict(tree_flatten(self.model.trainable_parameters()))
        expected = {f"{target}.lora_{matrix}" for target in self.targets for matrix in ("a", "b")}
        if set(parameters) != expected or not expected:
            raise RuntimeError("Only the selected LoRA matrices should be trainable")
        self.trainable_count = sum(value.size for value in parameters.values())
        self.sampling_key = mx.random.state[0]

    @staticmethod
    def load_checkpoint(location: Path):
        return load(str(location))

    @classmethod
    def validate_scoring(cls, gradient_checkpointing: bool, score_block_size: int) -> None:
        if gradient_checkpointing and cls.checkpoint_layers is None:
            raise ValueError(f"{cls.backend} does not support --gradient-checkpointing")
        if score_block_size and cls.cached_gradient is None:
            raise ValueError(f"{cls.backend} does not support --score-block-size")

    def render_prompt(self, messages, tools, **template_options) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True, **template_options
        )
        return self.tokenizer.encode(rendered, add_special_tokens=False)

    def toy_prompt_ids(self, messages: list[dict]) -> list[int]:
        return self.prompt_ids(messages, None, "none")

    def toy_answer(self, tokens: tuple[int, ...]) -> str:
        message = self.parse_completion(tokens, [])
        return (message["content"] or "").strip().lower()

    def is_stop_token(self, token: int) -> bool:
        return token in self.tokenizer.eos_token_ids

    def sample(self, prompt: list[int], limit: int, temperature: float) -> Turn:
        if not prompt or limit < 1 or not isfinite(temperature) or temperature <= 0:
            raise ValueError(
                "Sampling needs a prompt, positive token limit and positive temperature"
            )

        def sampler(logprobs):
            self.sampling_key, key = mx.random.split(self.sampling_key)
            return mx.random.categorical(logprobs / temperature, key=key)

        tokens, logprobs = [], []
        request = MODEL_REQUEST.get()
        progress_callback = None
        if request is not None:
            # generate_step builds a fresh cache when called without prompt_cache.
            started = monotonic()
            request.prefill_reused_tokens = 0

            def progress_callback(processed, total):
                request.prefill_processed_tokens = processed
                if processed == total:
                    request.prefill_to_first_token_seconds = monotonic() - started

        stream = generate_step(
            mx.array(prompt),
            self.model,
            max_tokens=limit,
            sampler=sampler,
            stream=mx.default_stream(mx.default_device()),
            prefill_step_size=self.prefill_step_size,
            prompt_progress_callback=progress_callback,
        )
        try:
            for token, raw_logprobs in stream:
                tokens.append(token)
                scaled = raw_logprobs / temperature
                logprobs.append((scaled[token] - mx.logsumexp(scaled)).item())
                if self.is_stop_token(token):
                    break
        finally:
            if request is not None:
                request.observed_sampled_tokens = len(tokens)
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
        cache = make_prompt_cache(self.model)
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
        (directory / "adapter_metadata.json").write_text(
            json.dumps(self.adapter_metadata, indent=2) + "\n", encoding="utf-8"
        )

    def load_adapter(self, directory: Path) -> None:
        metadata = directory / "adapter_metadata.json"
        if metadata.exists():
            if json.loads(metadata.read_text(encoding="utf-8")) != self.adapter_metadata:
                raise ValueError("Adapter metadata does not match the selected backend and model")
        elif not self.allow_legacy_adapter:
            raise ValueError("Adapter is missing base model metadata")
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

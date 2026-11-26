"""LoRA and chat support for looped Inkling 2 Mini checkpoints."""

from contextlib import contextmanager
from functools import partial
from math import isfinite

import mlx.core as mx
from mlx.nn.utils import checkpoint
from mlx_lm.models.inkling2 import layers

from .base import MLXPolicy
from .inkling_chat import parse_completion
from .inkling_score import CachedScoringDrift, CachedTurn


@contextmanager
def checkpoint_decoder_layers():
    """Recompute each decoder visit during backward, including its LoRA parameters."""
    decoder = layers.DecoderLayer
    forward = decoder.__call__

    def checkpointed(layer, *args, **kwargs):
        bound_forward = partial(forward, layer)
        if kwargs.get("cache") is not None:
            return bound_forward(*args, **kwargs)
        return checkpoint(layer, bound_forward)(*args, **kwargs)

    # Inkling's sliding and full attention layers share this method. Keep the
    # patch scoped to gradient evaluation so generation uses the normal path.
    decoder.__call__ = checkpointed
    try:
        yield
    finally:
        decoder.__call__ = forward


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


class InklingPolicy(MLXPolicy):
    backend = "inkling"
    serving_name = "inkling-2-mini"
    allow_legacy_adapter = True
    checkpoint_layers = staticmethod(checkpoint_decoder_layers)

    def configure_model(self) -> tuple[str, ...]:
        if self.model.model_type != "inkling2_loop":
            raise ValueError("This backend expects a looped Inkling 2 Mini checkpoint")
        language_model = self.model.model.language_model
        if not language_model.args.loop_config:
            raise ValueError("This backend expects a looped Inkling 2 Mini checkpoint")

        # Use the MLX reference operations for both sampling and training.
        layers._use_kernels = lambda: False
        layers.TopKRouter.__call__ = differentiable_router
        core = language_model.args.loop_core
        return (f"model.language_model.layers.{core.stop - 1}.self_attn.o_proj",)

    @property
    def target(self) -> str:
        return self.targets[0]

    def prompt_ids(self, messages: list[dict], tools: list[dict] | None, effort: str) -> list[int]:
        return self.render_prompt(messages, tools, reasoning_effort=effort)

    def toy_prompt_ids(self, messages: list[dict]) -> list[int]:
        return super().toy_prompt_ids(messages) + self.tokenizer.encode(
            "<|message_model|><|content_text|>", add_special_tokens=False
        )

    def toy_answer(self, tokens: tuple[int, ...]) -> str:
        return self.decode(tokens).split("<|")[0].strip().lower()

    def parse_completion(self, tokens, tools, *, include_undeclared=False, effort="none") -> dict:
        return parse_completion(self.decode(tokens), tools, include_undeclared=include_undeclared)

    def cached_gradient(self, turn, advantage, block_size, max_logprob_drift):
        largest_drift = 0.0
        while True:
            try:
                loss, drift, grad = CachedTurn(self, turn, block_size).loss_and_grad(
                    advantage, max_logprob_drift
                )
                return loss, drift, grad, block_size, max(largest_drift, drift)
            except CachedScoringDrift as exc:
                if block_size == 1 or not isfinite(exc.drift):
                    raise
                largest_drift = max(largest_drift, exc.drift)
                block_size //= 2

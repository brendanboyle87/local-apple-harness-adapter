"""Differentiate Inkling turns with the sampler's prefill and blocked decode."""

from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from math import isfinite, isqrt

import mlx.core as mx
from mlx.nn.utils import checkpoint
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.inkling2 import layers

from .trace import Turn

_MAX_CACHE_CHECKPOINTS = 16


class CachedScoringDrift(RuntimeError):
    def __init__(self, drift: float, limit: float):
        super().__init__(f"Cached-prefill scoring drift {drift:.4f} nats exceeds {limit:.4f}")
        self.drift = drift


def _visit_state(visit):
    kv, convolution = visit.caches
    window, padding, lengths = convolution.state
    return kv.state, (tuple(window), padding, lengths)


def _set_visit_state(visit, kv, conv) -> None:
    # MLX-LM's KV caches replace slices of array wrappers in place.
    visit[0].state = tuple(mx.array(x) if isinstance(x, mx.array) else x for x in kv)
    visit[1].state = (list(conv[0]), conv[1], conv[2])


@contextmanager
def _checkpoint_cached_layers():
    original = layers.DecoderLayer.__call__

    def forward(layer, hidden, blocks, *, cache=None, conv_mask=None, messages=None):
        def transition(hidden, blocks, mask, messages, kv, conv):
            private = copy(cache)
            private.caches = tuple(copy(part) for part in cache.caches)
            _set_visit_state(private, kv, conv)
            result = original(
                layer, hidden, blocks, cache=private, conv_mask=mask, messages=messages
            )
            return result, _visit_state(private)

        result, (kv, conv) = checkpoint(layer, transition)(
            hidden, blocks, conv_mask, messages, *_visit_state(cache)
        )
        _set_visit_state(cache, kv, conv)
        return result

    layers.DecoderLayer.__call__ = forward
    try:
        yield
    finally:
        layers.DecoderLayer.__call__ = original


@dataclass(frozen=True)
class _CacheState:
    tensors: dict
    metadata: dict

    @classmethod
    def capture(cls, cache):
        tensors, metadata = {}, {}
        for name, value in tree_flatten([_visit_state(visit) for visit in cache]):
            collection = tensors if isinstance(value, mx.array) else metadata
            collection[name] = value
        return cls(tensors, metadata)

    def restore(self, cache, tensors) -> None:
        values = tree_unflatten(list((self.metadata | tensors).items()))
        for visit, (kv, conv) in zip(cache, values, strict=True):
            _set_visit_state(visit, kv, conv)


class CachedTurn:
    """Materialize cache boundaries and recompute each segment during backward."""

    def __init__(self, policy, turn: Turn, block_size: int):
        if block_size < 1:
            raise ValueError("Cached scoring requires a positive completion block size")
        self.model = policy.model
        self.parameters = tree_flatten(self.model.trainable_parameters(), destination={})
        self.turn = turn
        self.input_ids = mx.array(turn.input_ids[:-1])[None, :]
        start = len(turn.prompt_token_ids) - 1
        self.spans = [
            (first, min(first + policy.prefill_step_size, start), None)
            for first in range(0, start, policy.prefill_step_size)
        ]
        for offset in range(0, len(turn.completion_token_ids), block_size):
            targets = turn.completion_token_ids[offset : offset + block_size]
            self.spans.append((start + offset, start + offset + len(targets), targets))

    def _transition(self, state, cache_tensors, span):
        first, end, targets = span
        model = self.model
        cache = model.make_cache()
        state.restore(cache, cache_tensors)
        logits = model(self.input_ids[:, first:end], cache=cache)[0]
        if targets is None:
            logprobs = None
        else:
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            logprobs = logprobs / self.turn.temperature
            chosen = mx.take_along_axis(logprobs, mx.array(targets)[:, None], axis=-1)[:, 0]
            logprobs = chosen - mx.logsumexp(logprobs, axis=-1)
        return _CacheState.capture(cache), logprobs

    def _vjp(self, before, after, span, cache_grad, advantage):
        parameters = self.parameters
        names = list(parameters)
        cache_names = list(before.tensors)
        count = len(names)
        targets = span[2]

        def forward(*inputs):
            weights = list(zip(names, inputs[:count], strict=True))
            tensors = dict(zip(cache_names, inputs[count:], strict=True))
            self.model.update(tree_unflatten(weights))
            observed, scores = self._transition(before, tensors, span)
            values = list(observed.tensors.values())
            return values if targets is None else [*values, scores]

        cotangents = [
            cache_grad[name] if name in cache_grad else mx.zeros_like(array)
            for name, array in after.tensors.items()
        ]
        if targets is not None:
            # At the on-policy step, d(-A * exp(logp - stop_gradient(logp)))/d(logp) = -A.
            cotangents.append(mx.full((len(targets),), -advantage, mx.float32))
        try:
            with _checkpoint_cached_layers():
                _, gradients = mx.vjp(
                    forward, [*parameters.values(), *before.tensors.values()], cotangents
                )
        finally:
            self.model.update(tree_unflatten(list(parameters.items())))
        return gradients[:count], dict(zip(cache_names, gradients[count:], strict=True))

    def _forward(self, stride):
        state = _CacheState.capture(self.model.make_cache())
        checkpoints, scores = [], []
        for index, span in enumerate(self.spans):
            if index % stride == 0:
                checkpoints.append((index, state))
            state, logprobs = self._transition(state, state.tensors, span)
            mx.eval(state.tensors, logprobs)
            if logprobs is not None:
                scores.append(logprobs)
        return checkpoints, mx.concatenate(scores)

    def logprobs(self) -> mx.array:
        return self._forward(len(self.spans))[1]

    def loss_and_grad(self, advantage: float, max_logprob_drift: float):
        # Keep at most 16 forward checkpoints. Subdivide large segments
        # during backward so long one-token fallbacks fit in unified memory.
        count = len(self.spans)
        stride = max(
            isqrt(count - 1) + 1, (count + _MAX_CACHE_CHECKPOINTS - 1) // _MAX_CACHE_CHECKPOINTS
        )
        checkpoints, logprobs = self._forward(stride)
        drift = mx.max(mx.abs(logprobs - mx.array(self.turn.old_logprobs))).item()
        if not isfinite(drift) or drift > max_logprob_drift:
            raise CachedScoringDrift(drift, max_logprob_drift)

        totals = [mx.zeros(weight.shape, dtype=mx.float32) for weight in self.parameters.values()]
        cache_grad = {}

        def advance(state, span):
            state, _ = self._transition(state, state.tensors, span)
            mx.eval(state.tensors)
            return state

        def backward(first, end, state):
            nonlocal totals, cache_grad
            if end - first > _MAX_CACHE_CHECKPOINTS:
                middle = (first + end) // 2
                midpoint = state
                for span in self.spans[first:middle]:
                    midpoint = advance(midpoint, span)
                backward(middle, end, midpoint)
                del midpoint
                backward(first, middle, state)
                return

            segment = self.spans[first:end]
            states = [state]
            for span in segment:
                state = advance(state, span)
                states.append(state)
            for index in reversed(range(len(segment))):
                weights, cache_grad = self._vjp(
                    states[index], states[index + 1], segment[index], cache_grad, advantage
                )
                totals = [
                    total + value.astype(mx.float32)
                    for total, value in zip(totals, weights, strict=True)
                ]
                mx.eval(totals, cache_grad)

        while checkpoints:
            first, state = checkpoints.pop()
            backward(first, min(first + stride, count), state)
        gradient = tree_unflatten(list(zip(self.parameters, totals, strict=True)))
        return -advantage * len(self.turn.completion_token_ids), drift, gradient

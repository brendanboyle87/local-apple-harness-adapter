"""Read-only derivative of the archived LFM3 sampler's prefill and one-token decode."""

from math import sqrt
import json

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten


BLOCK = 24


def same_wrapper(value):
    return mx.array(value) if isinstance(value, mx.array) else value


def snapshot(cache):
    # Cache slice assignment mutates array wrappers. Keep distinct wrappers.
    return tree_map(same_wrapper, [part.state for part in cache])


def restore(model, state):
    cache = model.make_cache()
    for part, saved in zip(cache, tree_map(same_wrapper, state), strict=True):
        part.state = saved
    return cache


def split(state):
    arrays = {}
    metadata = {}
    for name, value in tree_flatten(state):
        (arrays if isinstance(value, mx.array) else metadata)[name] = value
    return arrays, metadata


def log_probability(logits, token, temperature):
    normalized = (logits - mx.logsumexp(logits, keepdims=True)).squeeze(0)
    scaled = normalized / temperature
    return (scaled[token] - mx.logsumexp(scaled)).astype(mx.float32)


def transition(model, cache, inputs, span, temperature, materialize):
    first, end, tokens = span
    if tokens is None:
        model(inputs[:, first:end], cache=cache)
        if materialize:
            mx.eval([part.state for part in cache])
        return None
    values = []
    for position, token in enumerate(tokens, first):
        logits = model(inputs[:, position : position + 1], cache=cache)[:, -1, :]
        value = log_probability(logits, token, temperature)
        if materialize:
            mx.eval(value)
        values.append(value)
    return mx.stack(values)


def make_inputs(policy, turn, block):
    start = len(turn.prompt_token_ids) - 1
    inputs = mx.array(turn.input_ids[:-1])[None]
    spans = [
        (pos, min(pos + policy.prefill_step_size, start), None)
        for pos in range(0, start, policy.prefill_step_size)
    ]
    for offset in range(0, len(turn.completion_token_ids), block):
        tokens = turn.completion_token_ids[offset : offset + block]
        spans.append((start + offset, start + offset + len(tokens), tokens))
    return inputs, spans


def segmented(policy, turn, advantage, block=BLOCK):
    model = policy.model
    parameters = dict(tree_flatten(model.trainable_parameters()))
    names = list(parameters)
    weights = list(parameters.values())
    inputs, spans = make_inputs(policy, turn, block)
    cache = model.make_cache()
    states = []
    expected_values = []
    for span in spans:
        saved = snapshot(cache)
        mx.eval(saved)
        states.append(saved)
        expected_values.append(transition(model, cache, inputs, span, turn.temperature, True))
    states.append(snapshot(cache))
    del cache

    totals = None
    downstream = None
    recomputation_drift = 0
    for index in reversed(range(len(spans))):
        span = spans[index]
        before, metadata = split(states[index])
        after, _ = split(states[index + 1])
        before_names = list(before)
        after_names = list(after)
        propagate = downstream is not None
        has_scores = span[2] is not None
        assert has_scores or propagate
        if propagate:
            assert downstream.keys() == after.keys()
        cotangents = [downstream[k] for k in after_names] if propagate else []
        if has_scores:
            cotangents.append(mx.full((len(span[2]),), -advantage, dtype=mx.float32))
        previous_state = metadata

        def forward(*primals):
            model.update(tree_unflatten(list(zip(names, primals[: len(names)], strict=True))))
            mapping = previous_state | dict(zip(before_names, primals[len(names) :], strict=True))
            cache = restore(model, tree_unflatten(list(mapping.items())))
            scores = transition(model, cache, inputs, span, turn.temperature, False)
            if propagate:
                result, _ = split(snapshot(cache))
                assert result.keys() == after.keys()
                values = [result[k] for k in after_names]
            else:
                values = []
            if has_scores:
                values.append(scores)
            return values

        try:
            values, derivatives = mx.vjp(forward, [*weights, *before.values()], cotangents)
            # Materialize before restoring parameters and releasing this segment.
            mx.eval(values, derivatives)
        finally:
            model.update(tree_unflatten(list(parameters.items())))
        if has_scores:
            drift = mx.max(mx.abs(values[-1] - expected_values[index])).item()
            recomputation_drift = max(recomputation_drift, drift)
        downstream = dict(zip(before_names, derivatives[len(names) :], strict=True))
        contribution = derivatives[: len(names)]
        totals = (
            contribution
            if totals is None
            else [x + y for x, y in zip(totals, contribution, strict=True)]
        )
        mx.eval(totals)
        states.pop()
    forward = mx.concatenate([v for v in expected_values if v is not None])
    drift = mx.max(mx.abs(forward - mx.array(turn.old_logprobs))).item()
    return tree_unflatten(list(zip(names, totals, strict=True))), drift, recomputation_drift


def verify_reference(policy, turn):
    inputs, spans = make_inputs(policy, turn, BLOCK)
    advantage = 0.339163885587297

    def loss(model):
        cache = model.make_cache()
        values = []
        for span in spans:
            vector = transition(model, cache, inputs, span, turn.temperature, False)
            if vector is not None:
                values.append(vector)
        selected = mx.concatenate(values)
        return -advantage * mx.sum(selected), selected

    (_, scores), direct = nn.value_and_grad(policy.model, loss)(policy.model)
    mx.eval(direct, scores)
    reference = dict(tree_flatten(direct))
    comparisons = []
    for block in (8, BLOCK):
        result, drift, recomputed = segmented(policy, turn, advantage, block)
        values = dict(tree_flatten(result))
        norm = sum(mx.sum(x.astype(mx.float32) ** 2).item() for x in reference.values())
        difference = sum(
            mx.sum((values[k] - x).astype(mx.float32) ** 2).item() for k, x in reference.items()
        )
        relative = sqrt(difference / norm)
        comparisons.append(
            {
                "block": block,
                "relative_l2_vs_direct": relative,
                "archive_drift": drift,
                "recomputed_drift": recomputed,
            }
        )
        assert relative < 0.015, comparisons
        assert drift < 1e-4 and recomputed < 1e-4, comparisons
    print(json.dumps({"cached_reference_validation": comparisons}), flush=True)
    mx.clear_cache()


def cached_score(policy):
    def score(turn, advantage):
        gradient, forward_drift, recomputation_drift = segmented(policy, turn, advantage)
        return gradient, dict(forward_drift=forward_drift, recomputation_drift=recomputation_drift)

    return score

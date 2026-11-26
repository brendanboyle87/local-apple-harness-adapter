from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache  # noqa: E402

from apple_harness_adapter.backends.inkling_score import CachedScoringDrift, CachedTurn  # noqa: E402
from apple_harness_adapter.trace import Turn  # noqa: E402


class CumulativePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.array(0.07)

    @staticmethod
    def make_cache():
        return [CacheList(KVCache(), ArraysCache(1))]

    def __call__(self, ids, cache=None):
        previous = 0.0 if cache is None or cache[0][0].keys is None else cache[0][0].keys
        signal = mx.cumsum(ids * self.weight, axis=-1) + previous
        if cache is not None:
            cache[0][0].keys = signal[:, -1:]
            cache[0][0].offset += ids.shape[1]
        return mx.stack([signal, -signal], axis=-1)


@pytest.mark.parametrize(("block", "repeats"), [(1, 1), (3, 1), (1, 40)])
def test_cache_backward_propagates_through_recomputed_prefix_and_blocks(block, repeats):
    model = CumulativePolicy()
    policy = SimpleNamespace(model=model, prefill_step_size=2)
    prompt = (1, 2, 3, 1, 2, 3)
    completion = (1, 1, 0, 1, 0, 1, 0, 1) * repeats
    temperature = 0.8
    advantage = 0.7

    def dense_scores():
        logits = model(mx.array((prompt + completion)[:-1])[None, :])[0, len(prompt) - 1 :]
        scaled = logits / temperature
        chosen = mx.take_along_axis(scaled, mx.array(completion)[:, None], axis=-1)[:, 0]
        return chosen - mx.logsumexp(scaled, axis=-1)

    turn = Turn(prompt, completion, tuple(dense_scores().tolist()), temperature)
    _, reference = nn.value_and_grad(model, lambda: -advantage * dense_scores().sum())()
    result, drift, gradient = CachedTurn(policy, turn, block).loss_and_grad(advantage, 0.001)
    assert result == pytest.approx(-advantage * len(completion))
    assert drift < 0.001
    assert gradient["weight"].item() == pytest.approx(reference["weight"].item(), rel=1e-5)
    assert model.weight.item() == pytest.approx(0.07)

    invalid = Turn(prompt, completion, (-20.0,) * len(completion), temperature)
    with pytest.raises(CachedScoringDrift):
        CachedTurn(policy, invalid, block).loss_and_grad(advantage, 0.001)

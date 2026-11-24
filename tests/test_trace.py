import pytest

from apple_harness_adapter.trace import Rollout, Turn, group_advantages


def test_turn_preserves_the_prompt_boundary():
    turn = Turn((0, 1), (2, 3), (-0.3, -1.2), 0.8)
    assert turn.input_ids == (0, 1, 2, 3)
    assert turn.completion_mask == (0, 0, 1, 1)


@pytest.mark.parametrize("logprobs", [(0.01,), (float("nan"),), (-0.5, -0.4)])
def test_turn_rejects_invalid_behavior_probabilities(logprobs):
    with pytest.raises(ValueError):
        Turn((0,), (1,), logprobs, 0.8)


def test_group_uses_correctness_and_success_only_efficiency():
    turn = Turn((0,), (1,), (-0.1,), 0.8)
    failure = Rollout("task", (turn,), 0, 0)
    slower_success = Rollout("task", (turn,), 1, 15)
    fast_success = Rollout("task", (turn,), 1, 1)
    assert failure.reward == 0
    assert slower_success.reward == 1
    assert fast_success.reward == pytest.approx(1 + 0.1 * 14 / 15)
    assert group_advantages([failure, failure]) == [0, 0]
    advantages = group_advantages([failure, slower_success, fast_success])
    assert advantages[0] < 0 < advantages[1] < advantages[2]
    assert sum(advantages) == pytest.approx(0)
    with pytest.raises(ValueError, match="same task"):
        group_advantages([failure, Rollout("other", (turn,), 0, 0)])

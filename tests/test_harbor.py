import pytest

pytest.importorskip("openenv")
pytest.importorskip("harbor")

from openenv.harbor.models import HarborRolloutResult, HarborTurn  # noqa: E402

from apple_harness_adapter.harbor import training_rollout  # noqa: E402


def test_harbor_bonus_counts_invocations_instead_of_available_tools():
    turn = HarborTurn(
        turn=0,
        prompt_token_ids=[1, 2],
        completion_token_ids=[3],
        per_token_logps=[-0.5],
        loss_mask=[0, 0, 1],
        sampling_params={"temperature": 0.8},
        n_tools=4,
        tool_calls=[{"name": "write"}],
    )
    result = HarborRolloutResult(task_name="example", reward=1, turns=[turn], n_trainable_tokens=1)
    rollout = training_rollout(result)
    assert rollout.tool_calls == 1
    assert rollout.reward == pytest.approx(1 + 0.1 * 14 / 15)
    assert rollout.turns[0].completion_mask == (0, 0, 1)

    turn.loss_mask = [0, 1, 0]
    with pytest.raises(RuntimeError, match="partial completion mask"):
        training_rollout(result)

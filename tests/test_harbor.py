import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("openenv")
pytest.importorskip("harbor")

from openenv.harbor.models import HarborRolloutResult, HarborTurn  # noqa: E402

from apple_harness_adapter.endpoint import function_call  # noqa: E402
from apple_harness_adapter.harbor import collect_harbor, training_rollout  # noqa: E402
from apple_harness_adapter.trace import Turn  # noqa: E402


def test_harbor_bonus_counts_invocations_instead_of_available_tools():
    turn = HarborTurn(
        turn=0,
        prompt_token_ids=[1, 2],
        completion_token_ids=[3],
        per_token_logps=[-0.5],
        loss_mask=[0, 0, 1],
        sampling_params={"temperature": 0.8},
        n_tools=4,
        text="[safeagents] model-authored text is not a verdict",
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


@pytest.mark.skipif(os.getenv("APPLE_HARNESS_DOCKER_TEST") != "1", reason="requires Docker")
def test_opencode_executes_a_tool_and_reconciles_local_tokens(tmp_path):
    task = Path(__file__).resolve().parents[1] / "examples/harbor/write-answer-opencode"

    def prompt_ids(messages, tools, _effort):
        if not tools:
            return [91, 92]
        if any(message["role"] == "tool" for message in messages):
            return [11, 22, 31, 32, 23, 24]
        return [11, 22]

    def sample(prompt, _limit, temperature):
        return Turn(tuple(prompt), (31, 32), (-0.1, -0.2), temperature)

    def parse_completion(_tokens, tools, *, include_undeclared, effort):
        if not tools:
            return {"role": "assistant", "content": "Write answer"}
        if not policy.wrote_answer:
            policy.wrote_answer = True
            tool = next(item["function"] for item in tools if item["function"]["name"] == "write")
            properties = tool["parameters"]["properties"]
            path = "filePath" if "filePath" in properties else "path"
            call = function_call(
                "write", {path: "/workspace/answer.txt", "content": "harbor ready\n"}
            )
            return {"role": "assistant", "content": None, "tool_calls": [call]}
        return {"role": "assistant", "content": "Done."}

    policy = SimpleNamespace(
        model_name="harness-fixture",
        prompt_ids=prompt_ids,
        sample=sample,
        parse_completion=parse_completion,
        is_stop_token=lambda token: token == 32,
        wrote_answer=False,
    )
    args = SimpleNamespace(
        harness="opencode",
        safe_agents="enforce",
        attempts=1,
        reward_key="",
        max_turns=5,
        max_output=128,
        max_context=4096,
        effort="none",
        temperature=0.8,
    )
    records = []
    (rollout,) = asyncio.run(
        collect_harbor(policy, args, tmp_path, lambda _index, record: records.append(record), task)
    )
    (record,) = records
    assert record["atif"] == "match" and record["ok"]
    assert rollout.correctness == 1.0 and rollout.tool_calls == 1
    assert rollout.turns[0].completion_token_ids == (31, 32)
    assert rollout.turns[0].old_logprobs == pytest.approx((-0.1, -0.2))

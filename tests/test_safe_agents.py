import asyncio
import json
import os
from pathlib import Path
from shutil import copytree
from types import SimpleNamespace

import pytest

pytest.importorskip("safeagents")

from atif_scan import Status  # noqa: E402
from safeagents.guard import Config  # noqa: E402

from apple_harness_adapter.safe_agents import SafetyAudit, TrainingScanner  # noqa: E402
from apple_harness_adapter.trace import Rollout, Turn, group_advantages  # noqa: E402

WORKSPACE_ANSWER = "/workspace/answer.txt"
REWARD_FILE = "/logs/verifier/reward.txt"
ANSWER = "harbor ready\n"
REQUEST = {
    "messages": [{"role": "user", "content": "Write the answer in the workspace."}],
    "tools": [{"type": "function", "function": {"name": "write"}}],
}


def write_call(call_id, path, content):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "write", "arguments": json.dumps({"path": path, "content": content})},
    }


def response(*calls):
    return {
        "id": "test-completion",
        "prompt_token_ids": [1, 2],
        "choices": [
            {
                "message": {"role": "assistant", "content": None, "tool_calls": list(calls)},
                "finish_reason": "tool_calls",
                "token_ids": [3, 4],
                "logprobs": {"content": [{"logprob": -1}, {"logprob": -2}]},
            }
        ],
    }


def test_guard_keeps_allowed_action_and_policy_tokens_in_a_mixed_response():
    allowed = write_call("allowed", WORKSPACE_ANSWER, ANSWER)
    denied = write_call("denied", REWARD_FILE, "guard test canary\n")
    payload = response(allowed, denied)
    audit = SafetyAudit("enforce")
    asyncio.run(audit.review_response(REQUEST, payload, Config(scanner=TrainingScanner())))

    choice = payload["choices"][0]
    assert choice["message"]["tool_calls"] == [allowed]
    assert choice["finish_reason"] == "tool_calls"
    assert "tamper.reward_write" in choice["message"]["content"]
    assert payload["prompt_token_ids"] == [1, 2]
    assert choice["token_ids"] == [3, 4]
    assert choice["logprobs"] == {"content": [{"logprob": -1}, {"logprob": -2}]}
    assert [call["call_id"] for call in audit.calls] == ["allowed", "denied"]
    assert [call["verdict"] for call in audit.calls] == ["passed", "blocked"]
    assert audit.blocked_tool_calls == 1


@pytest.mark.parametrize("mode", ["enforce", "log-only"])
def test_guard_ends_all_blocked_responses_only_in_enforce_mode(mode):
    payload = response(write_call("denied", REWARD_FILE, "guard test canary\n"))
    audit = SafetyAudit(mode)
    asyncio.run(audit.review_response(REQUEST, payload, Config(scanner=TrainingScanner())))
    choice = payload["choices"][0]
    if mode == "enforce":
        assert "tool_calls" not in choice["message"]
        assert choice["finish_reason"] == "stop"
        assert audit.blocked_tool_calls == 1
    else:
        assert choice["message"]["tool_calls"][0]["id"] == "denied"
        assert audit.calls[0]["verdict"] == "flagged"
        assert audit.blocked_tool_calls == 0


def test_enforced_block_overrides_a_successful_verifier_and_efficiency_bonus():
    turn = Turn((1,), (2,), (-1.0,), 0.8)
    clean = Rollout("fixture", (turn,), 1.0, 1)
    blocked = Rollout("fixture", (turn,), 1.0, 1, blocked_tool_calls=1)
    assert clean.reward > 1 and blocked.reward == 0 and blocked.correctness == 1.0
    clean_advantage, blocked_advantage = group_advantages([clean, blocked])
    assert blocked_advantage < 0 < clean_advantage


def test_tool_that_was_not_declared_is_blocked():
    payload = response(write_call("undeclared", WORKSPACE_ANSWER, ANSWER))
    audit = SafetyAudit("enforce")
    asyncio.run(
        audit.review_response(REQUEST | {"tools": []}, payload, Config(scanner=TrainingScanner()))
    )
    assert audit.blocked_tool_calls == 1
    assert "tool_calls" not in payload["choices"][0]["message"]


def test_guard_surfaces_detector_errors(monkeypatch):
    scanner = TrainingScanner()
    error = SimpleNamespace(
        spec=SimpleNamespace(id="broken-detector"), result=SimpleNamespace(status=Status.ERROR)
    )
    monkeypatch.setattr(scanner.engine, "evaluate", lambda *_: [error])
    payload = response(write_call("allowed", WORKSPACE_ANSWER, ANSWER))
    with pytest.raises(RuntimeError, match="broken-detector"):
        asyncio.run(
            SafetyAudit("enforce").review_response(REQUEST, payload, Config(scanner=scanner))
        )


def test_failed_guard_invalidates_the_attempt_even_if_the_verifier_passes(monkeypatch, tmp_path):
    from openenv.harbor.models import HarborRolloutResult

    from apple_harness_adapter import harbor

    def fail(*_args):
        raise RuntimeError("fixture scan error")

    async def rollout(**kwargs):
        client = kwargs["inference"]._client
        for _ in range(2):
            reply = await client.post("/chat/completions", json=REQUEST)
            assert reply.status_code == 500
            assert "choices" not in reply.json()
        return HarborRolloutResult(trial_name="failing-guard", reward=1.0)

    monkeypatch.setattr(TrainingScanner, "signals", fail)
    monkeypatch.setattr(harbor, "run_rollout", rollout)
    monkeypatch.setattr(
        harbor,
        "chat_completion",
        lambda *_args, **_kwargs: response(write_call("allowed", WORKSPACE_ANSWER, ANSWER)),
    )
    options = SimpleNamespace(
        safe_agents="enforce",
        attempts=1,
        reward_key="",
        max_turns=3,
        max_output=128,
        max_context=4096,
        effort="none",
        temperature=0.8,
    )
    records = []
    with pytest.raises(RuntimeError, match="SafeAgents failed.*fixture scan error"):
        asyncio.run(
            harbor.collect_harbor(
                SimpleNamespace(model_name="guard-fixture"),
                options,
                tmp_path,
                lambda _index, record: records.append(record),
                tmp_path,
            )
        )
    (record,) = records
    assert not record["ok"]
    assert record["safe_agents"]["error"] == "RuntimeError: fixture scan error"
    assert record["reward"] == 1.0 and record["effective_reward"] is None


@pytest.mark.skipif(os.getenv("APPLE_HARNESS_DOCKER_TEST") != "1", reason="requires Docker")
@pytest.mark.parametrize("context_stop", [False, True])
def test_real_pi_executes_allowed_call_and_reconciles_blocked_call(tmp_path, context_stop):
    from apple_harness_adapter.backends.inkling_chat import parse_completion
    from apple_harness_adapter.harbor import collect_harbor

    source = Path(__file__).resolve().parents[1] / "examples/harbor/write-answer"
    task = tmp_path / "task"
    copytree(source, task)
    verifier = task / "tests/test.sh"
    verifier.write_text(
        verifier.read_text().replace(
            "if [ ! -L /workspace/answer.txt ]",
            f"if [ ! -e {REWARD_FILE} ] && [ ! -L /workspace/answer.txt ]",
        )
    )

    def inkling_write(path, content):
        invocation = {"name": "write", "args": {"path": path, "content": content}}
        return (
            "<|message_model|>write<|content_invoke_tool_json|>"
            + json.dumps(invocation)
            + "<|end_message|>"
        )

    completions = [
        inkling_write(WORKSPACE_ANSWER, ANSWER) + inkling_write(REWARD_FILE, "guard test canary\n"),
        "<|message_model|><|content_text|>Done.<|end_message|>",
    ]
    completions = iter(text + "<|content_model_end_sampling|>" for text in completions)
    prompts = []

    def prompt_ids(messages, _tools, _effort):
        prompts.append(messages)
        return [11, 22] if len(prompts) == 1 else [11, 22, 31, 32, 23, 24]

    policy = SimpleNamespace(
        model_name="inkling-2-mini",
        prompt_ids=prompt_ids,
        is_stop_token=lambda token: token == 32,
        sample=lambda ids, _limit, temp: Turn(tuple(ids), (31, 32), (-0.1, -0.2), temp),
        parse_completion=lambda _tokens, tools, *, include_undeclared, effort: parse_completion(
            next(completions), tools, include_undeclared=include_undeclared
        ),
    )
    options = SimpleNamespace(
        safe_agents="enforce",
        attempts=1,
        reward_key="",
        max_turns=3,
        max_output=128,
        max_context=6 if context_stop else 4096,
        effort="none",
        temperature=0.8,
    )
    records = []
    (rollout,) = asyncio.run(
        collect_harbor(
            policy,
            options,
            tmp_path / "trials",
            lambda _index, record: records.append(record),
            task,
        )
    )

    (record,) = records
    assert record["atif"] == "match" and record["ok"]
    assert record["budget_stop_count"] == int(context_stop)
    assert len(rollout.turns) == (1 if context_stop else 2)
    assert record["reward"] == 1.0 and record["effective_reward"] == 0
    assert rollout.tool_calls == 1 and rollout.blocked_tool_calls == 1
    assert rollout.turns[0].completion_token_ids == (31, 32)
    assert rollout.turns[0].old_logprobs == pytest.approx((-0.1, -0.2))
    assert "[safeagents]" in record["turns"][0]["text"]
    assert "[safeagents]" in json.dumps(prompts[1])
    assert [event["verdict"] for event in record["safe_agents"]["calls"]] == ["passed", "blocked"]

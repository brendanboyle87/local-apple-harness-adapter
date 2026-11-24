import json
from pathlib import Path
from shutil import copytree
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
pytest.importorskip("openenv")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from apple_harness_adapter import checkpoint, harbor  # noqa: E402
from apple_harness_adapter.cli import harbor_run  # noqa: E402
from apple_harness_adapter.trace import Rollout, Turn  # noqa: E402


class FixturePolicy:
    target = "projection"

    def __init__(self):
        self.model = nn.Module()
        self.model.projection = nn.Module()
        self.model.projection.lora_b = mx.zeros((1,))
        self.sampling_key = mx.random.key(42)

    @staticmethod
    def score(model, turn):
        direction = 1 if turn.completion_token_ids == (1,) else -1
        return -1 + direction * model.projection.lora_b

    def save(self, path):
        path.mkdir()
        mx.save_safetensors(str(path / "weights.safetensors"), {"b": self.model.projection.lora_b})

    def load_adapter(self, path):
        self.model.projection.lora_b = mx.load(str(path / "weights.safetensors"))["b"]


@pytest.mark.parametrize("interrupt_at", ["rollout", "commit"])
def test_resume_retries_partial_group_with_same_adam_rng_and_schedule(
    tmp_path, monkeypatch, interrupt_at
):
    task_source = Path(__file__).resolve().parents[1] / "examples/harbor/write-answer"
    tasks = [tmp_path / name for name in ("first-task", "second-task")]
    for task in tasks:
        copytree(task_source, task)

    interrupted = False

    async def collect(policy, _args, trials, save_attempt, task):
        rollouts = []
        for index in range(2):
            policy.sampling_key, key = mx.random.split(policy.sampling_key)
            sampled = mx.random.randint(0, 1_000_000, key=key).item()
            turn = Turn((42,), (index + 1,), (-1.0,), 0.8)
            turn = Turn(
                turn.prompt_token_ids,
                turn.completion_token_ids,
                (policy.score(policy.model, turn).item(),),
                turn.temperature,
            )
            save_attempt(index, {"draw": sampled, "task": task.name})
            if interrupted and interrupt_at == "rollout" and trials.parent.name == "group-0003":
                raise KeyboardInterrupt
            rollouts.append(Rollout(task.name, (turn,), float(task == tasks[0] and index == 0), 1))
        return rollouts

    write = checkpoint._write_committed_json

    def interrupt_commit(path, document):
        if interrupted and interrupt_at == "commit" and path.parent.name == "group-0003":
            raise KeyboardInterrupt
        write(path, document)

    monkeypatch.setattr(harbor, "collect_harbor", collect)
    monkeypatch.setattr(checkpoint, "_write_committed_json", interrupt_commit)
    args = SimpleNamespace(
        learning_rate=1e-5,
        max_logprob_drift=0.2,
        gradient_checkpointing=False,
        output=tmp_path / "continuous",
        task=tasks,
        adapter=None,
        resume=False,
        steps=3,
        collect_only=False,
        safe_agents="enforce",
    )
    harbor_run(FixturePolicy(), args)
    reference = args.output

    args.output = tmp_path / "resumed"
    interrupted = True
    with pytest.raises(KeyboardInterrupt):
        harbor_run(FixturePolicy(), args)
    partial = args.output / "group-0003"
    assert (partial / "attempt-01.json").exists()
    assert (partial / "training.safetensors").exists() == (interrupt_at == "commit")
    assert not (partial / "metrics.json").exists()

    interrupted = False
    args.resume = True
    groups = harbor_run(FixturePolicy(), args)["groups"]
    assert [group["task"] for group in groups] == ["first-task", "second-task", "first-task"]
    assert [group["updated"] for group in groups] == [True, False, True]

    (abandoned,) = (args.output / "interrupted").iterdir()
    assert (abandoned / "attempt-01.json").read_bytes() == (
        partial / "attempt-01.json"
    ).read_bytes()
    for group in ("group-0001", "group-0002", "group-0003"):
        for name in ("attempt-01.json", "attempt-02.json"):
            assert (args.output / group / name).read_bytes() == (
                reference / group / name
            ).read_bytes()
        expected = mx.load(str(reference / group / "training.safetensors"))
        actual = mx.load(str(args.output / group / "training.safetensors"))
        assert actual.keys() == expected.keys()
        assert all(mx.array_equal(value, expected[key]).item() for key, value in actual.items())
    assert actual["step"].item() == 2
    expected_weights = mx.load(str(reference / partial.name / "adapter/weights.safetensors"))
    restored_weights = mx.load(str(partial / "adapter/weights.safetensors"))
    assert mx.array_equal(expected_weights["b"], restored_weights["b"]).item()

    recorded_metrics = json.loads((partial / "metrics.json").read_text())
    assert harbor_run(FixturePolicy(), args)["groups"][-1] == recorded_metrics
    args.stage = "changed"
    with pytest.raises(ValueError, match="configuration differs"):
        harbor_run(FixturePolicy(), args)

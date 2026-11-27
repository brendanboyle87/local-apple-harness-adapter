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
from mlx.optimizers import Adam  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from apple_harness_adapter import checkpoint, harbor  # noqa: E402
from apple_harness_adapter.backends.base import MLXPolicy  # noqa: E402
from apple_harness_adapter.cli import harbor_run  # noqa: E402
from apple_harness_adapter.grpo import GRPO  # noqa: E402
from apple_harness_adapter.trace import Rollout, Turn  # noqa: E402


class FixturePolicy(MLXPolicy):
    backend = "inkling"
    model_identity = {"model": "fixture", "revision": "test"}
    targets = ("projection",)

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


def test_resume_treats_legacy_harbor_runs_as_pi(tmp_path):
    config = {"checkpoint_version": checkpoint.CHECKPOINT_VERSION, "steps": 1, "backend": "lfm3"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert checkpoint.prepare_run(tmp_path, config | {"harness": "pi"}, True, None, None) == []
    with pytest.raises(ValueError, match="Resume configuration differs for harness"):
        checkpoint.prepare_run(tmp_path, config | {"harness": "opencode"}, True, None, None)


@pytest.mark.parametrize("legacy", [False, True])
def test_collection_resume_retries_partial_group_without_adam(tmp_path, monkeypatch, legacy):
    task_source = Path(__file__).resolve().parents[1] / "examples/harbor/write-answer"
    tasks = [tmp_path / name for name in ("first-task", "second-task")]
    for task in tasks:
        copytree(task_source, task)

    async def collect(policy, _args, trials, save_attempt, task):
        policy.sampling_key, key = mx.random.split(policy.sampling_key)
        draw = mx.random.randint(0, 1_000_000, key=key).item()
        correctness = float(task == tasks[0])
        save_attempt(
            0,
            {
                "draw": draw,
                "reward": correctness,
                "efficiency": {"mlx_group_peak_to_date_bytes": 0, "logical_tokens": 6},
            },
        )
        return [Rollout(task.name, (), correctness, 1)]

    def unexpected_trainer(*args, **kwargs):
        pytest.fail("Collection initialized a trainer")

    monkeypatch.setattr(GRPO, "__init__", unexpected_trainer)
    monkeypatch.setattr(harbor, "collect_harbor", collect)
    args = SimpleNamespace(
        learning_rate=1e-5,
        max_logprob_drift=0.2,
        gradient_checkpointing=False,
        score_block_size=0,
        output=tmp_path / "continuous",
        task=tasks,
        adapter=None,
        resume=False,
        steps=3,
        collect_only=True,
        safe_agents="enforce",
    )
    reference = args.output
    expected = harbor_run(FixturePolicy(), args)["groups"]

    write = checkpoint._write_committed_json

    def interrupt_commit(path, document):
        if path.parent.name == "group-0002":
            raise KeyboardInterrupt
        write(path, document)

    args.output = tmp_path / "resumed"
    with monkeypatch.context() as patch:
        patch.setattr(checkpoint, "_write_committed_json", interrupt_commit)
        with pytest.raises(KeyboardInterrupt):
            harbor_run(FixturePolicy(), args)
    partial = args.output / "group-0002"
    assert not (partial / "metrics.json").exists()
    assert (partial / checkpoint.SAMPLING_FILE).is_file()

    first = args.output / "group-0001"
    if legacy:
        sampling_file = first / checkpoint.SAMPLING_FILE
        old_state = mx.load(str(sampling_file))
        adam = Adam(args.learning_rate)
        adam.init(FixturePolicy().model.trainable_parameters())
        old_state.update(tree_flatten(adam.state))
        mx.save_safetensors(str(first / checkpoint.STATE_FILE), old_state)
        sampling_file.unlink()

    args.resume = True
    with pytest.raises(ValueError, match="Resume configuration differs for collect_only"):
        checkpoint.prepare_run(
            args.output,
            checkpoint.read_config(args.output) | {"collect_only": False},
            True,
            None,
            None,
        )
    resumed_policy = FixturePolicy()
    actual = harbor_run(resumed_policy, args)["groups"]
    assert [group["task"] for group in actual] == ["first-task", "second-task", "first-task"]
    for key in ("correctness", "efficiency", "updated"):
        assert [group[key] for group in actual] == [group[key] for group in expected]
    (abandoned,) = (args.output / "interrupted").iterdir()
    assert (abandoned / "attempt-01.json").read_bytes() == (
        partial / "attempt-01.json"
    ).read_bytes()
    for index in range(1, 4):
        group = f"group-{index:04d}"
        resumed = args.output / group
        assert (resumed / "attempt-01.json").read_bytes() == (
            reference / group / "attempt-01.json"
        ).read_bytes()
        if legacy and index == 1:
            continue
        assert not (resumed / checkpoint.STATE_FILE).exists()
        state = mx.load(str(resumed / checkpoint.SAMPLING_FILE))
        assert state.keys() == {"sampling_key"}
        expected_state = mx.load(str(reference / group / checkpoint.SAMPLING_FILE))
        assert mx.array_equal(state["sampling_key"], expected_state["sampling_key"]).item()
    assert mx.array_equal(resumed_policy.sampling_key, state["sampling_key"]).item()


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
        score_block_size=0,
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
    config_path = args.output / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("backend")
    config.pop("model_identity")
    config_path.write_text(json.dumps(config))
    assert harbor_run(FixturePolicy(), args)["groups"][-1] == recorded_metrics
    different_backend = FixturePolicy()
    different_backend.backend = "qwen3"
    with pytest.raises(ValueError, match="configuration differs for backend"):
        harbor_run(different_backend, args)
    args.stage = "changed"
    with pytest.raises(ValueError, match="configuration differs"):
        harbor_run(FixturePolicy(), args)

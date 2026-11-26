"""Group-boundary checkpoints for the sequential Harbor runner."""

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

CHECKPOINT_VERSION = 1
STATE_FILE = "training.safetensors"


def _write_committed_json(path: Path, document: dict) -> None:
    pending = path.with_name(f".{path.name}.pending")
    with pending.open("w", encoding="utf-8") as out:
        json.dump(document, out, indent=2, allow_nan=False)
        out.write("\n")
    pending.replace(path)


def read_config(path: Path) -> dict:
    return json.loads((path / "config.json").read_text(encoding="utf-8"))


@contextmanager
def lock_run(output: Path, resume: bool):
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"A process already owns the run at {output}") from exc
        yield


def prepare_run(output: Path, config: dict, resume: bool, policy, optimizer) -> list[dict]:
    if not resume:
        _write_committed_json(output / "config.json", config)
        return []

    previous = read_config(output)
    if previous.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("This run predates resumable checkpoints")
    if "harness" in config:
        previous.setdefault("harness", "pi")
    if "backend" not in previous and "model_identity" not in previous:
        previous["backend"] = "inkling"
        # Older runs already compared the requested model and revision below.
        if config.get("backend") == "inkling":
            previous["model_identity"] = config.get("model_identity")
    changed = sorted(
        key
        for key in previous.keys() | config.keys()
        if key != "steps" and previous.get(key) != config.get(key)
    )
    if changed:
        raise ValueError(f"Resume configuration differs for {', '.join(changed)}")

    groups = sorted(output.glob("group-[0-9]*"), key=lambda group: int(group.name[6:]))
    metrics = []
    latest_state = latest_adapter = unfinished = None
    for index, group in enumerate(groups, start=1):
        if group.name != f"group-{index:04d}" or not group.is_dir():
            raise ValueError(f"Unexpected group in run directory: {group}")
        result_file = group / "metrics.json"
        if not result_file.exists():
            if index != len(groups):
                raise ValueError(f"Later groups follow the unfinished group at {group}")
            unfinished = group
            break
        if not (group / STATE_FILE).is_file():
            raise ValueError(f"Missing training state for completed group at {group}")
        result = json.loads(result_file.read_text(encoding="utf-8"))
        metrics.append(result)
        latest_state = group / STATE_FILE
        if result["updated"]:
            latest_adapter = group / "adapter"

    if len(metrics) > config["steps"]:
        raise ValueError("Target steps is less than the number of completed groups")
    if latest_state is not None:
        _restore_state(latest_state, policy, optimizer, sum(row["updated"] for row in metrics))
    if latest_adapter is not None:
        policy.load_adapter(latest_adapter)
    if unfinished is not None and len(metrics) < config["steps"]:
        archive = output / "interrupted"
        archive.mkdir(exist_ok=True)
        unfinished.rename(archive / f"{unfinished.name}-{uuid4().hex}")
    return metrics


def commit_group(group: Path, result: dict, policy, optimizer) -> None:
    if result["updated"]:
        policy.save(group / "adapter")
    state = dict(tree_flatten(optimizer.state))
    state["sampling_key"] = policy.sampling_key
    mx.save_safetensors(str(group / STATE_FILE), state)
    # The atomic metrics rename commits the group after its tensors exist.
    _write_committed_json(group / "metrics.json", result)


def _restore_state(path: Path, policy, optimizer, updates: int) -> None:
    state = mx.load(str(path))
    key = state.pop("sampling_key", None)
    expected = dict(tree_flatten(optimizer.state))
    if (
        key is None
        or key.shape != policy.sampling_key.shape
        or key.dtype != policy.sampling_key.dtype
        or state.keys() != expected.keys()
        or any(
            value.shape != expected[name].shape or value.dtype != expected[name].dtype
            for name, value in state.items()
        )
        or state["step"].item() != updates
        or state["learning_rate"].item() != expected["learning_rate"].item()
        or not all(mx.all(mx.isfinite(value)).item() for value in state.values())
    ):
        raise ValueError(f"Invalid checkpoint state at {path}")
    optimizer.state = tree_unflatten(list(state.items()))
    policy.sampling_key = key

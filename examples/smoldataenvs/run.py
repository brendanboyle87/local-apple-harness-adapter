"""Train or evaluate with the staged SmolDataEnvs recipe."""

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, quantiles
from time import monotonic
from types import SimpleNamespace
from uuid import uuid4

from apple_harness_adapter.backends import DEFAULT_MODELS, DEFAULT_REVISIONS
from apple_harness_adapter.telemetry import observed_sum

MODEL = DEFAULT_MODELS["lfm3"]
REVISION = DEFAULT_REVISIONS[MODEL]
ROLLOUT = {
    "max_turns": 10,
    "max_output": 1024,
    "max_context": 6144,
    "effort": "low",
    "reward_key": "correctness,reward",
    "safe_agents": "enforce",
}
EVAL_SEED = 7297
BOOTSTRAP_SEED = 8921
BOOTSTRAP_RESAMPLES = 10000


def save_json(path, data):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def train(args, manifest):
    seed, steps = {"warmup": (81, 3), "train": (1229, 50)}[manifest["split"]]
    command = [
        sys.executable,
        "-m",
        "apple_harness_adapter.cli",
        "--backend",
        "lfm3",
        "--revision",
        REVISION,
        "--seed",
        str(seed),
        "harbor",
        "--harness",
        manifest["harness"],
        "--steps",
        str(steps if args.steps is None else args.steps),
        "--output",
        str(args.output),
    ]
    for row in manifest["tasks"]:
        command.extend(("--task", str(args.tasks / row["path"])))
    for key, value in ROLLOUT.items():
        command.extend((f"--{key.replace('_', '-')}", str(value)))
    if args.adapter:
        command.extend(("--adapter", str(args.adapter)))
    if args.resume:
        command.append("--resume")
    return subprocess.run(command).returncode


def score(policy, harness, path, output, seed, adapter):
    import mlx.core as mx

    from apple_harness_adapter.cli import harbor_run

    measurement = output / "measurement.json"
    if measurement.is_file():
        return json.loads(measurement.read_text())
    if output.exists():
        output.rename(output.with_name(f"{output.name}-interrupted-{uuid4().hex[:8]}"))
    started = monotonic()
    policy.sampling_key = mx.random.key(seed)
    options = SimpleNamespace(
        backend="lfm3",
        model=MODEL,
        revision=REVISION,
        rank=32,
        temperature=0.8,
        seed=seed,
        harness=harness,
        command="harbor",
        learning_rate=1e-5,
        max_logprob_drift=0.2,
        gradient_checkpointing=False,
        score_block_size=0,
        output=output,
        task=[path],
        adapter=adapter,
        resume=False,
        steps=1,
        attempts=1,
        collect_only=True,
        **ROLLOUT,
    )
    try:
        (result,) = harbor_run(policy, options)["groups"]
        outcome = {
            "reward": result["correctness"][0],
            "calls": result["tool_calls"][0],
            "timed_out": False,
        }
        measurements = result.get("efficiency")
        if measurements is not None:
            (outcome["efficiency"],) = measurements
    except RuntimeError:
        trace = output / "group-0001/attempt-01.json"
        if not trace.is_file():
            raise
        record = json.loads(trace.read_text())
        if record.get("exception_type") != "AgentTimeoutError" or record["safe_agents"]["error"]:
            raise
        calls = sum(
            len(turn["tool_calls"])
            for turn in record["turns"]
            if turn["role"] == "agent" and not turn["discarded"]
        )
        outcome = {"reward": record["reward"] or 0.0, "calls": calls, "timed_out": True}
        if "efficiency" in record:
            outcome["efficiency"] = {
                key: value for key, value in record["efficiency"].items() if key != "tool_spans"
            }
    outcome["task_seconds"] = monotonic() - started
    save_json(measurement, outcome)
    return outcome


def source_resamples(sources):
    rng = random.Random(BOOTSTRAP_SEED)
    for _ in range(BOOTSTRAP_RESAMPLES):
        yield rng.choices(sources, k=len(sources))


def interval95(values):
    if len(values) < 2:
        return None
    boundaries = quantiles(values, n=40)
    return [boundaries[0], boundaries[-1]]


def summarize(rows):
    groups = defaultdict(lambda: [0, 0, 0, 0])
    base_passes = adapter_passes = joint = 0
    for row in rows:
        base, adapter = row["base"], row["adapter"]
        base_pass = base["reward"] == 1
        adapter_pass = adapter["reward"] == 1
        base_passes += base_pass
        adapter_passes += adapter_pass
        count = groups[row["source"]]
        count[0] += 1
        count[1] += adapter_pass - base_pass
        if base_pass and adapter_pass:
            joint += 1
            count[2] += base["calls"]
            count[3] += adapter["calls"]
    sources = list(groups.values())
    _, _, base_calls, adapter_calls = map(sum, zip(*sources))
    accuracy, calls = [], []
    for sample in source_resamples(sources):
        n, delta, before, after = map(sum, zip(*sample))
        accuracy.append(100 * delta / n)
        if before:
            calls.append(100 * (after / before - 1))

    return {
        "tasks": len(rows),
        "source_datasets": len(sources),
        "base_passes": base_passes,
        "adapter_passes": adapter_passes,
        "pass_rate_change_points": 100 * (adapter_passes - base_passes) / len(rows),
        "pass_rate_change_ci95": interval95(accuracy),
        "shared_successes": joint,
        "base_calls_on_shared_successes": base_calls,
        "adapter_calls_on_shared_successes": adapter_calls,
        "tool_call_change_percent": 100 * (adapter_calls / base_calls - 1) if base_calls else None,
        "tool_call_change_ci95": interval95(calls),
    }


def resource_value(arm, key):
    if key == "task_seconds":
        return arm.get(key)
    if key == "tool_calls":
        return arm["calls"]
    return arm.get("efficiency", {}).get(key)


def paired_resource_changes(group, key, samples, additive):
    changes = {"mean_change_percent": None, "mean_change_ci95": None}
    if additive:
        changes |= {"per_success_change_percent": None, "per_success_change_ci95": None}
    sources = defaultdict(lambda: [0, 0, 0, 0])
    for row in group:
        before, after = (resource_value(row[arm], key) for arm in ("base", "adapter"))
        if before is None or after is None:
            return changes
        values = sources[row["source"]]
        values[0] += before
        values[1] += after
        values[2] += row["base"]["reward"] == 1
        values[3] += row["adapter"]["reward"] == 1
    if not sources:
        return changes

    def relative(before, after):
        return 100 * (after / before - 1)

    before, after, base_passes, adapter_passes = map(sum, zip(*sources.values()))
    if not before:
        return changes
    ratios, per_success = [], []
    for sample in samples:
        b, a, bp, ap = map(sum, zip(*(sources.get(source, (0, 0, 0, 0)) for source in sample)))
        if b:
            ratios.append(relative(b, a))
            if additive and bp and ap:
                per_success.append(relative(b / bp, a / ap))
    changes |= {
        "mean_change_percent": relative(before, after),
        "mean_change_ci95": interval95(ratios),
    }
    if additive and base_passes and adapter_passes:
        changes |= {
            "per_success_change_percent": relative(before / base_passes, after / adapter_passes),
            "per_success_change_ci95": interval95(per_success),
        }
    return changes


def summarize_measured_resources(rows):
    """Aggregate new evaluation telemetry with explicit coverage for resumed older runs."""
    additive = (
        "task_seconds",
        "relay_requests",
        "backend_requests",
        "generation_requests",
        "agent_turns",
        "tool_calls",
        "pi_tool_calls",
        "tool_executions",
        "tool_execution_seconds",
        "rendered_prompt_tokens",
        "prompt_tokens",
        "completion_tokens",
        "logical_tokens",
        "reasoning_tokens",
        "prefill_processed_tokens",
        "prefill_reused_tokens",
        "prefill_to_first_token_seconds",
        "decode_seconds",
        "inference_seconds",
        "generation_relay_seconds",
        "endpoint_seconds",
    )
    peaks = ("mlx_attempt_peak_bytes", "mlx_max_request_peak_bytes")
    sources = list(dict.fromkeys(row["source"] for row in rows))
    samples = list(source_resamples(sources)) if sources else []
    summary = {
        "schema_version": 1,
        "bootstrap": {
            "unit": "source_dataset",
            "sources": len(sources),
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
        },
    }
    for cohort, group in (
        ("all_tasks", rows),
        (
            "shared_successes",
            [row for row in rows if row["base"]["reward"] == row["adapter"]["reward"] == 1],
        ),
    ):
        arms = {}
        for arm in ("base", "adapter"):
            passes = sum(row[arm]["reward"] == 1 for row in group)
            metrics = {}
            for key in additive + peaks:
                values = [resource_value(row[arm], key) for row in group]
                observed = sum(value is not None for value in values)
                complete = bool(values) and observed == len(values)
                total = observed_sum(values) if key in additive else None
                metrics[key] = {
                    "observed_attempts": observed,
                    "mean": mean(values) if complete else None,
                    "median": median(values) if complete else None,
                    "max": max(values) if complete else None,
                    "total": total,
                    "per_success": total / passes if total is not None and passes else None,
                }
            arms[arm] = {"passes": passes, "metrics": metrics}
        summary[cohort] = {
            "tasks": len(group),
            "arms": arms,
            "paired": {
                key: paired_resource_changes(group, key, samples, key in additive)
                for key in additive + peaks
            },
        }
    return summary


def evaluate(args, manifest):
    import mlx.core as mx
    from harbor.publisher.packager import Packager
    from mlx.utils import tree_flatten

    from apple_harness_adapter.backends.lfm3 import LFM3Policy

    tasks = manifest["tasks"][: args.limit]
    with (args.adapter / "adapters.safetensors").open("rb") as weights:
        digest = hashlib.file_digest(weights, "sha256").hexdigest()
    plan = {
        "tasks": tasks,
        "harness": manifest["harness"],
        "model": MODEL,
        "revision": REVISION,
        "adapter_sha256": digest,
        "seed_start": EVAL_SEED,
        "rollout": ROLLOUT,
        "task_digests": [
            Packager.compute_content_hash(args.tasks / task["path"])[0] for task in tasks
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    config = args.output / "config.json"
    if config.is_file() and json.loads(config.read_text()) != plan:
        raise ValueError(
            "Saved evaluation has a different task list, adapter, or rollout configuration"
        )
    save_json(config, plan)
    policy = LFM3Policy(MODEL, REVISION, 32, EVAL_SEED)
    base = tree_flatten(policy.model.trainable_parameters())
    policy.load_adapter(args.adapter)
    arms = {
        "base": (base, None),
        "adapter": (tree_flatten(policy.model.trainable_parameters()), args.adapter),
    }
    rows = []
    for index, task in enumerate(tasks):
        result = {"task": task["name"], "source": task["source"]}
        order = ("base", "adapter") if index % 2 == 0 else ("adapter", "base")
        for arm in order:
            weights, adapter = arms[arm]
            policy.model.load_weights(weights, strict=False)
            mx.eval(policy.model.trainable_parameters())
            result[arm] = score(
                policy,
                manifest["harness"],
                args.tasks / task["path"],
                args.output / task["name"] / arm,
                EVAL_SEED + index,
                adapter,
            )
        rows.append(result)
        save_json(args.output / "pairs.json", rows)
        print(f"Evaluated {len(rows)}/{len(tasks)} tasks", flush=True)
    summary = summarize(rows)
    summary["efficiency"] = summarize_measured_resources(rows)
    save_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--steps", type=int)
    training.add_argument("--resume", action="store_true")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--limit", type=int)
    for command in (training, evaluation):
        command.add_argument(
            "--tasks", type=Path, required=True, help="output directory from prepare.py"
        )
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--adapter", type=Path, required=command is evaluation)
    args = parser.parse_args()
    size = args.steps if args.command == "train" else args.limit
    if size is not None and size < 1:
        parser.error("--steps and --limit must be positive")
    manifest = json.loads((args.tasks / "manifest.json").read_text())
    allowed = ("warmup", "train") if args.command == "train" else ("heldout",)
    if manifest["split"] not in allowed or not manifest["tasks"]:
        parser.error(f"{args.command} requires prepared tasks from {', '.join(allowed)}")
    if args.command == "train":
        return train(args, manifest)
    from apple_harness_adapter.checkpoint import lock_run

    args.output.mkdir(parents=True, exist_ok=True)
    with lock_run(args.output, resume=True):
        return evaluate(args, manifest)


if __name__ == "__main__":
    raise SystemExit(main())

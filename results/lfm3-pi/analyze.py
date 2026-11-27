"""Regenerate the recorded three-arm evaluation and base/local efficiency summaries."""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECIPE = HERE.parents[1] / "examples" / "smoldataenvs"
sys.path.insert(0, str(RECIPE))

from efficiency import extract, read_json, read_jsonl, summarize_efficiency  # noqa: E402
from run import BOOTSTRAP_RESAMPLES, summarize  # noqa: E402


def comparison(rows, reference, adapted):
    # The original verifier produced binary scores in all three arms.
    pairs = []
    for row in rows:
        pair = {"source": row["bucket_prefix"]}
        for label, arm in (("base", reference), ("adapter", adapted)):
            measurement = row["arms"][arm]
            if measurement["reward"] not in (0, 1):
                raise ValueError(f"Expected binary verifier scores for {row['task']}")
            pair[label] = {"reward": measurement["reward"], "calls": measurement["tool_calls"]}
        pairs.append(pair)

    stats = summarize(pairs)
    shared = stats["shared_successes"]
    arms = (reference, adapted)
    return {
        "pairs": stats["tasks"],
        "reference_arm": reference,
        "adapter_arm": adapted,
        "passes": {reference: stats["base_passes"], adapted: stats["adapter_passes"]},
        "adapter_only": stats["adapter_passes"] - shared,
        "reference_only": stats["base_passes"] - shared,
        "joint_passes": shared,
        "calls_on_joint_passes": {
            reference: stats["base_calls_on_shared_successes"],
            adapted: stats["adapter_calls_on_shared_successes"],
        },
        "timeouts": {arm: sum(row["arms"][arm]["agent_timed_out"] for row in rows) for arm in arms},
        "blocked_calls": {
            arm: sum(row["arms"][arm]["blocked_calls"] for row in rows) for arm in arms
        },
        "pass_rate_change_pp": stats["pass_rate_change_points"],
        "joint_tool_call_change_percent": stats["tool_call_change_percent"],
        "paired_uncertainty": {
            "pass_rate_change_pp_ci95": stats["pass_rate_change_ci95"],
            "joint_tool_call_change_percent_ci95": stats["tool_call_change_ci95"],
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_unit": "source_dataset",
            "source_datasets": stats["source_datasets"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check the regenerated artifacts")
    parser.add_argument(
        "--efficiency", action="store_true", help="analyze released efficiency data"
    )
    parser.add_argument("--archive", type=Path, help="extract from the original lfm3-150 directory")
    parser.add_argument("--tokenizer-snapshot", type=Path, help="local pinned LFM3 tokenizer files")
    parser.add_argument(
        "--write", action="store_true", help="write regenerated efficiency artifacts"
    )
    args = parser.parse_args()
    if bool(args.archive) != bool(args.tokenizer_snapshot):
        parser.error("--archive and --tokenizer-snapshot must be supplied together")
    if args.write and (args.check or not (args.efficiency or args.archive)):
        parser.error("--write requires an efficiency analysis without --check")
    rows = read_jsonl(HERE / "pairs.jsonl")
    arms = ("base", "published", "local")
    comparisons = {
        f"{adapted}_vs_{reference}": comparison(rows, reference, adapted)
        for reference, adapted in (("base", "local"), ("base", "published"), ("published", "local"))
    }
    summary = {
        "completed_tasks": len(rows),
        "rollouts": len(rows) * len(arms),
        "comparisons": comparisons,
        "source_datasets": len({row["bucket_prefix"] for row in rows}),
        "passes": {arm: sum(row["arms"][arm]["reward"] == 1 for row in rows) for arm in arms},
    }
    if args.check:
        if summary != json.loads((HERE / "summary.json").read_text()):
            parser.exit(1, "Regenerated results differ from summary.json.\n")
        observed = json.loads((RECIPE / "observed.json").read_text())
        if comparisons["local_vs_base"] != observed["evaluation"]:
            parser.exit(1, "The local comparison differs from observed.json.\n")
        print(f"Verified {len(rows)} pairs and all three comparisons.")
    elif not (args.efficiency or args.archive):
        print(json.dumps(summary, indent=2))
    if args.efficiency or args.archive:
        efficiency(args, rows, comparisons["local_vs_base"])


def efficiency(args, rows, original):
    from efficiency import plot_efficiency

    measured_path = HERE / "efficiency-pairs.jsonl"
    measured = (
        extract(args.archive, args.tokenizer_snapshot, rows, read_json(HERE / "provenance.json"))
        if args.archive
        else read_jsonl(measured_path)
    )
    results = summarize_efficiency(measured, rows)
    joint_calls = results["shared_successes"]["metrics"]["tool_calls"]
    if (
        joint_calls["mean_change_percent"] != original["joint_tool_call_change_percent"]
        or joint_calls["mean_change_percent_ci95"]
        != original["paired_uncertainty"]["joint_tool_call_change_percent_ci95"]
    ):
        raise ValueError("Efficiency analysis differs from the original joint tool-call result")
    artifacts = {
        HERE / "efficiency.json": json.dumps(results, indent=2, allow_nan=False) + "\n",
        HERE / "efficiency.svg": plot_efficiency(results),
    }
    if args.archive:
        artifacts[measured_path] = "".join(
            json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in measured
        )
    if args.check:
        for path, contents in artifacts.items():
            if path.read_text(encoding="utf-8") != contents:
                raise ValueError(f"Regenerated efficiency artifact differs from {path.name}")
        print(f"Verified archived efficiency comparisons for {len(measured)} pairs.")
    elif args.write:
        for path, contents in artifacts.items():
            path.write_text(contents, encoding="utf-8")
    else:
        print(artifacts[HERE / "efficiency.json"], end="")


if __name__ == "__main__":
    main()

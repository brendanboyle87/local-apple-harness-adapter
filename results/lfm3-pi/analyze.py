"""Regenerate the recorded three-arm evaluation summary from paired measurements."""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECIPE = HERE.parents[1] / "examples" / "smoldataenvs"
sys.path.insert(0, str(RECIPE))

from run import summarize  # noqa: E402


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
            "bootstrap_resamples": 10000,
            "bootstrap_unit": "source_dataset",
            "source_datasets": stats["source_datasets"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check both recorded summaries")
    args = parser.parse_args()
    rows = [json.loads(line) for line in (HERE / "pairs.jsonl").read_text().splitlines()]
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
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

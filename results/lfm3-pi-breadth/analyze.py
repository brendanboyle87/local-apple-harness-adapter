"""Regenerate the broader LFM3 Pi evaluation from archived measurements."""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lfm3-pi"))

from analyze import (  # noqa: E402
    add_efficiency_arguments,
    comparison,
    publish_efficiency,
    validate_efficiency_arguments,
    verify_joint_calls,
)
from efficiency import extract, plot_efficiency, read_json, read_jsonl, summarize_efficiency  # noqa: E402

REFERENCE = HERE.parent / "lfm3-pi"
REFERENCES = ("local", "base", "published")
ARMS = ("base", "local", "broad", "published")
MANIFESTS = HERE.parents[1] / "examples/smoldataenvs"
VERIFIER_LOG = "group-0001/trials/*/verifier/test-stdout.txt"
GRADER_SCORE = re.compile(r"\[grader\] gold=.* pred=.* (?:reward|correctness)=([01]\.0) method=\w+")
MISSING_ANSWER = {
    "[grader] no answer at /workdir/answer.txt",
    "[grader] no answer at /workdir/answer.txt -> all rewards 0",
}


def overlap_sensitivity(rows):
    training_inputs = defaultdict(list)
    training_sources = set()
    for split in ("warmup", "breadth"):
        for task in read_jsonl(MANIFESTS / f"{split}.jsonl"):
            training_sources.add(task["prefix"])
            for entry in task["inputs"]:
                training_inputs[entry["sha256"]].append({"split": split, "task": task["name"]})

    heldout = read_jsonl(MANIFESTS / "heldout.jsonl")
    if [task["name"] for task in heldout] != [row["task"] for row in rows]:
        raise ValueError("Paired task order differs from the frozen held-out manifest")
    excluded = {}
    for task in heldout:
        shared = {
            entry["sha256"]: training_inputs[entry["sha256"]]
            for entry in task["inputs"]
            if entry["sha256"] in training_inputs
        }
        if shared:
            excluded[task["name"]] = shared
    return {
        "overlapping_source_names": sorted(training_sources & {task["prefix"] for task in heldout}),
        "excluded_tasks": excluded,
        "broad_vs_base": comparison(
            [row for row in rows if row["task"] not in excluded], "base", "broad"
        ),
    }


def verifier_outcomes(rows, runs):
    reference = runs / "lfm3-150/evaluation"
    breadth = runs / "lfm3-breadth/evaluation"
    outcomes = {arm: {"pass": 0, "wrong_answer": 0, "missing_answer": 0} for arm in ARMS}
    joined = set()
    for row in rows:
        for arm in ARMS:
            archive = breadth if arm == "broad" else reference
            logs = list((archive / row["task"] / arm).glob(VERIFIER_LOG))
            if len(logs) != 1 or logs[0] in joined:
                raise ValueError(f"Expected one unique verifier log for {row['task']} / {arm}")
            log = logs[0]
            joined.add(log)
            lines = [line for line in log.read_text().splitlines() if line.startswith("[grader]")]
            if len(lines) != 1:
                raise ValueError(f"Expected one grader outcome at {log}")
            (line,) = lines
            if line in MISSING_ANSWER:
                reward, outcome = 0.0, "missing_answer"
            else:
                match = GRADER_SCORE.fullmatch(line)
                if not match:
                    raise ValueError(f"Unexpected grader outcome at {log}")
                reward = float(match[1])
                outcome = "pass" if reward == 1 else "wrong_answer"
            if reward != row["arms"][arm]["reward"]:
                raise ValueError(f"Verifier and published reward disagree at {log}")
            outcomes[arm][outcome] += 1

    if joined != {
        log for archive in (reference, breadth) for log in archive.glob(f"*/*/{VERIFIER_LOG}")
    }:
        raise ValueError("The verifier archives contain attempts outside the released pairs")
    return {"joined_logs": len(joined), "arms": outcomes}


def screening_budget(runs):
    counts = {}
    for phase in ("lfm3-followup", "lfm3-overnight", "lfm3-breadth"):
        rows = read_json(runs / phase / "calibration/results.json")
        counts[phase] = sum(len(row["scores"]) for row in rows)
    return {"attempts_by_archive": counts, "total_attempts": sum(counts.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check the recorded summary")
    add_efficiency_arguments(parser)
    parser.add_argument("--reference-archive", type=Path, help="original lfm3-150 archive")
    parser.add_argument("--audit-runs", type=Path, help="original runs directory for release audit")
    args = parser.parse_args()
    validate_efficiency_arguments(parser, args)
    if args.reference_archive and not args.archive:
        parser.error("--reference-archive requires --archive")
    rows = read_jsonl(HERE / "pairs.jsonl")
    summary = {
        "completed_tasks": len(rows),
        "new_rollouts": len(rows),
        "comparisons": {f"broad_vs_{arm}": comparison(rows, arm, "broad") for arm in REFERENCES},
        "source_datasets": len({row["bucket_prefix"] for row in rows}),
        "passes": {arm: sum(row["arms"][arm]["reward"] == 1 for row in rows) for arm in ARMS},
    }
    audit = {"overlap_sensitivity": overlap_sensitivity(rows)}
    if args.audit_runs:
        audit["verifier_outcomes"] = verifier_outcomes(rows, args.audit_runs)
        audit["task_selection_screening"] = screening_budget(args.audit_runs)
    if args.check:
        if summary != json.loads((HERE / "summary.json").read_text()):
            parser.exit(1, "Regenerated results differ from summary.json.\n")
        published_audit = read_json(HERE / "release-audit.json")
        if any(result != published_audit[key] for key, result in audit.items()):
            parser.exit(1, "Regenerated release audit differs from release-audit.json.\n")
        print(f"Verified {len(rows)} tasks and all three broader-adapter comparisons.")
        print("Verified the input-overlap sensitivity analysis.")
        if args.audit_runs:
            print(
                f"Verified all {audit['verifier_outcomes']['joined_logs']} verifier logs and screening."
            )
    elif args.audit_runs:
        print(json.dumps(audit, indent=2, allow_nan=False))
    elif not (args.efficiency or args.archive):
        print(json.dumps(summary, indent=2))
    if args.efficiency or args.archive:
        efficiency(args, rows, summary["comparisons"])


def efficiency(args, rows, original):
    reference = read_jsonl(REFERENCE / "efficiency-pairs.jsonl")
    if args.archive:
        broad = extract(
            args.archive,
            args.tokenizer_snapshot,
            rows,
            read_json(HERE / "provenance.json"),
            archive_key="lfm3-breadth",
            archive_arms=("broad",),
            output_arms=("broad",),
        )
        previous = extract(
            args.reference_archive or args.archive.with_name("lfm3-150"),
            args.tokenizer_snapshot,
            read_jsonl(REFERENCE / "pairs.jsonl"),
            read_json(REFERENCE / "provenance.json"),
            output_arms=REFERENCES,
        )
        additional = []
        for current, archived, released in zip(broad, previous, reference, strict=True):
            if any(archived["arms"][arm] != released["arms"][arm] for arm in ("base", "local")):
                raise ValueError(f"Previously published efficiency differs for {archived['task']}")
            additional.append(
                current
                | {
                    "arms": current["arms"]
                    | {
                        "published": archived["arms"]["published"],
                    }
                }
            )
    else:
        additional = read_jsonl(HERE / "efficiency-pairs.jsonl")
    paired = []
    for added, prior in zip(additional, reference, strict=True):
        if any(added[key] != prior[key] for key in ("task", "bucket_prefix", "seed")):
            raise ValueError("Reference efficiency rows are out of order")
        if set(added["arms"]) != {"broad", "published"} or set(prior["arms"]) != {"base", "local"}:
            raise ValueError("Unexpected efficiency arms")
        paired.append(added | {"arms": prior["arms"] | added["arms"]})
    comparisons = {
        f"broad_vs_{arm}": summarize_efficiency(paired, rows, (arm, "broad")) for arm in REFERENCES
    }
    for name, result in comparisons.items():
        verify_joint_calls(result, original[name])
    figure = plot_efficiency(comparisons["broad_vs_base"], "30-task adapter")
    publish_efficiency(
        args, HERE, additional, {"schema_version": 1, "comparisons": comparisons}, figure
    )


if __name__ == "__main__":
    main()

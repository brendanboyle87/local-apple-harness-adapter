"""Regenerate the broader LFM3 Pi evaluation from archived measurements."""

import argparse
import json
import sys
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check the recorded summary")
    add_efficiency_arguments(parser)
    parser.add_argument("--reference-archive", type=Path, help="original lfm3-150 archive")
    args = parser.parse_args()
    validate_efficiency_arguments(parser, args)
    if args.reference_archive and not args.archive:
        parser.error("--reference-archive requires --archive")
    rows = read_jsonl(HERE / "pairs.jsonl")
    arms = ("base", "published", "local", "broad")
    summary = {
        "completed_tasks": len(rows),
        "new_rollouts": len(rows),
        "comparisons": {f"broad_vs_{arm}": comparison(rows, arm, "broad") for arm in REFERENCES},
        "source_datasets": len({row["bucket_prefix"] for row in rows}),
        "passes": {arm: sum(row["arms"][arm]["reward"] == 1 for row in rows) for arm in arms},
    }
    if args.check:
        if summary != json.loads((HERE / "summary.json").read_text()):
            parser.exit(1, "Regenerated results differ from summary.json.\n")
        print(f"Verified {len(rows)} tasks and all three broader-adapter comparisons.")
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

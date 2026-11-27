"""Regenerate the broader LFM3 Pi evaluation from archived measurements."""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lfm3-pi"))

from analyze import comparison  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check the recorded summary")
    args = parser.parse_args()
    rows = [json.loads(line) for line in (HERE / "pairs.jsonl").read_text().splitlines()]
    arms = ("base", "published", "local", "broad")
    references = ("local", "base", "published")
    summary = {
        "completed_tasks": len(rows),
        "new_rollouts": len(rows),
        "comparisons": {f"broad_vs_{arm}": comparison(rows, arm, "broad") for arm in references},
        "source_datasets": len({row["bucket_prefix"] for row in rows}),
        "passes": {arm: sum(row["arms"][arm]["reward"] == 1 for row in rows) for arm in arms},
    }
    if args.check:
        if summary != json.loads((HERE / "summary.json").read_text()):
            parser.exit(1, "Regenerated results differ from summary.json.\n")
        print(f"Verified {len(rows)} tasks and all three broader-adapter comparisons.")
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

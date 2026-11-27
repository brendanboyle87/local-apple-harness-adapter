# Recover the historical 30-task engine

The four-file patch reconstructs the archived engine from the source in the v0.1.0 release commit. It includes the archived checkpoint helper, CLI, Harbor integration, and relay. The manifest in `provenance.json` records the SHA-256 of each of the 18 engine files. [Historical source patch](historical-engine.patch) · [Recorded checksums](provenance.json)

Run from the repository root with full Git history available. The extracted source is written to a new temporary directory.

```sh
evidence_dir="$PWD/results/lfm3-pi-breadth"
historical_dir="$(mktemp -d)"
git archive ae278b016c9582584e5e2844da36eb13c02ea811 src/apple_harness_adapter \
  | tar -x -C "$historical_dir"
(
  cd "$historical_dir" &&
  git apply --unidiff-zero "$evidence_dir/historical-engine.patch"
)
python3 - "$evidence_dir/provenance.json" "$historical_dir/src" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

provenance, engine = map(Path, sys.argv[1:])
expected = json.loads(provenance.read_text())["historical_engine_files_sha256"]
if len(expected) != 18:
    raise SystemExit("Expected checksums for 18 source files")
actual = {
    file.relative_to(engine).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
    for file in (engine / "apple_harness_adapter").rglob("*")
    if file.is_file()
}
if actual != expected:
    bad = sorted(path for path in actual.keys() | expected.keys()
                 if actual.get(path) != expected.get(path))
    raise SystemExit("Historical engine checksum mismatch for " + ", ".join(bad))
print(f"Verified {len(actual)} historical engine files under {engine}")
PY
```

To use the historical package for a source audit, set `PYTHONPATH` to `$historical_dir/src` and use the dependency versions in the provenance record. The reconstruction covers engine source. The Python environment, model, adapters, and archived rollout data have their own provenance. [Historical run settings](provenance.json)

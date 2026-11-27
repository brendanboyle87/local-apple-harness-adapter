import json
from math import acos, degrees, sqrt
from pathlib import Path
from contextlib import ExitStack

import numpy as np
from safetensors import safe_open


p = Path(__file__).resolve().parent
labels = ("historical", "patched", "cached")
pairings = (("historical", "patched"), ("historical", "cached"), ("patched", "cached"))
values = {label: json.loads((p / f"{label}.json").read_text()) for label in labels}
for run in values.values():
    assert run["turns"] == 49 and run["completion_tokens"] == 7433
    assert run["adapter_sha256"] == values["historical"]["adapter_sha256"]

pairs = {}
with ExitStack() as stack:
    tensors = {
        label: stack.enter_context(safe_open(p / f"{label}.safetensors", framework="np"))
        for label in labels
    }
    keys = tensors["historical"].keys()
    assert len(keys) == 306 and all(v.keys() == keys for v in tensors.values())
    accum = {pair: np.zeros(4) for pair in pairings}
    grouped = {pair: {} for pair in pairings}
    for name in keys:
        vectors = {
            label: f.get_tensor(name).astype(np.float64).ravel() for label, f in tensors.items()
        }
        for a, b in pairings:
            x, y = vectors[a], vectors[b]
            difference = x - y
            statistics = np.array(
                [np.vdot(x, x), np.vdot(y, y), np.vdot(x, y), np.vdot(difference, difference)]
            )
            accum[a, b] += statistics
            layer = int(name.split(".")[2])
            grouped[a, b].setdefault(layer, np.zeros(4))[:] += statistics


def summarize(v):
    x, y, dot, dist = v
    cosine = max(-1.0, min(1.0, dot / sqrt(x * y)))
    return dict(
        first_norm=sqrt(x),
        reference_norm=sqrt(y),
        relative_l2=sqrt(dist / y),
        cosine=cosine,
        angle_degrees=degrees(acos(cosine)),
        absolute_l2=sqrt(dist),
    )


for pair, v in accum.items():
    rollup = summarize(v)
    worst = max(grouped[pair], key=lambda layer: summarize(grouped[pair][layer])["relative_l2"])
    rollup["highest_relative_error_layer"] = worst
    rollup["highest_relative_error_layer_metrics"] = summarize(grouped[pair][worst])
    pairs[f"{pair[0]}_vs_{pair[1]}"] = rollup

summary = {
    "runs": {
        label: {k: v for k, v in run.items() if k != "details"} for label, run in values.items()
    },
    "comparisons": pairs,
}
cached = values["cached"]["details"]
summary["cached_forward_max_drift"] = max(r["forward_drift"] for r in cached)
summary["cached_recomputation_max_drift"] = max(r["recomputation_drift"] for r in cached)
(p / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
print(json.dumps(summary, indent=2, allow_nan=False))

# Group 41 gradient measurements

The saved audit compares the same 49 retained turns and 7,433 completion tokens at the group 40 adapter weights. The aggregate summary includes cosine similarity, gradient norms, and relative L2 differences. The per-turn files include the recorded masks' token counts, advantages, rollout weights, and score drift checks. [Aggregate summary](summary.json) · [Historical dense](historical.json) · [Patched dense](patched.json) · [Sampling-shaped cached reference](cached.json) · [Single-turn direct backward check](cached-reference-validation.json)

The scripts require the original `runs/lfm3-breadth` archive, including group 41 traces, the group 40 adapter, and the archived engine. The audit checks the engine file hashes and adapter digest before loading the model. Follow the repository's memory preflight, then run from the repository root. Use the locked environment for MLX 0.32.4 and a separate environment with MLX 0.32.5 and the same remaining dependencies for the patched dense comparison. Set `patched_python` to that environment's Python executable. [Memory preflight](../../../AGENTS.md) · [Historical source reconstruction](../historical-engine.md)

```sh
audit_dir="$(mktemp -d)"
cp results/lfm3-pi-breadth/gradient-audit/*.py "$audit_dir/"
export PYTHONPATH="$PWD/runs/lfm3-breadth/engine"
.venv/bin/python "$audit_dir/audit.py" historical
"$patched_python" "$audit_dir/audit.py" patched
MLX_SDPA_VJP_FALLBACK=1 .venv/bin/python "$audit_dir/audit.py" cached
MLX_SDPA_VJP_FALLBACK=1 .venv/bin/python "$audit_dir/validate_peak.py"
.venv/bin/python "$audit_dir/compare.py"
```

The scripts write gradient tensors and regenerated JSON into the temporary directory. The released evidence contains numeric measurements and code.

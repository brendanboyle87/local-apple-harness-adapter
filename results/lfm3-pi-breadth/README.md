# LFM3-1.7B BF16 + Pi, 30 training tasks

The 30-task checkpoint from MLX Harness Adapter passed **100/150** held-out SmolDataEnvs tasks with LFM3-1.7B BF16 and Pi. The base model passed **72/150**. On this task selection the adapter had 35 adapter-only wins and seven base-only wins. The paired 95% intervals below describe variation across sampled task sources.

| Reference | Reference passes | 30-task passes | Paired pass-rate change (95% CI) | Shared successes | Tool-call change on shared successes (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 72 | 100 | +18.67 pp (+10.27 to +26.97) | 65 | −15.72% (−22.83% to −8.09%) |
| 14-task local adapter | 83 | 100 | +11.33 pp (+3.75 to +18.95) | 72 | +7.84% (+0.87% to +15.05%) |
| Published Harness Hub Pi adapter | 100 | 100 | 0.00 pp (−6.29 to +6.62) | 88 | +27.43% (+18.56% to +36.90%) |

The historical evaluation used one stochastic attempt per task and arm. The broader arm ran after the three reference arms on the same holdout with the same Pi version, verifier, limits, and per-task seed. Both the broader adapter and Harness Hub's adapter solved 100 tasks, with 12 unique passes each. The study covers one training run per schedule. The schedules differed in task mix and number of optimizer updates. [Earlier experiment's measurements](../lfm3-pi/pairs.jsonl)

Across all 150 tasks the 30-task adapter used **20.04% fewer logical tokens** than base and **42.43% fewer logical tokens per success**. The latter divides all logical prompt and completion tokens, including failed attempts, by the number of passes. Total elapsed time fell 1.16%, with a paired 95% interval of −7.73% to +6.49%. [Efficiency results, timing caveats, and reproduction](efficiency.md)

## Training budget

The checkpoint inherited 24 warmup rollouts in three groups on two tasks. Main training began from that adapter with fresh Adam and ran 400 rollouts in 50 groups on 30 other tasks. Forty-five groups applied updates. The main phase took 7.2 hours and peaked at 35.464 GiB in the MLX allocator on a 128 GB M4 Max. [Warmup provenance](../lfm3-pi/provenance.json#training) · [Main run provenance](provenance.json)

The archived breadth supervisor reused the follow-up and overnight selections and screened additional candidates. The `lfm3-followup`, `lfm3-overnight`, and `lfm3-breadth` calibration archives contain **64, 176, and 136 attempts**, respectively. These 376 attempts were task-selection work shared across experiments. Using the frozen schedule lets a reproduction skip screening. [Screening counts](release-audit.json) · [Frozen training manifest](../../examples/smoldataenvs/breadth.jsonl)

## Input-overlap sensitivity

The warmup and breadth training manifests have disjoint source names from the holdout. Joining their input SHA-256 values to the held-out manifest found two identical training input files in three held-out questions. The matching held-out task IDs are `0094_620_94620248_qa_2`, `0094_620_94620248_qa_3`, and `0076_340_76340399_qa_5`.

The historical result remains **100 versus 72 on 150 tasks**. The sensitivity analysis derives the exclusions from the hashes, keeps the order of the remaining pairs, and applies the same source-dataset bootstrap. It yields **97 versus 72 on 147 tasks**, with 32 adapter-only wins and seven base-only wins. The gain is **17.01 percentage points** (95% interval **+8.82 to +25.00**) across 91 source clusters. [Matching hashes and full paired result](release-audit.json)

## What the verifier recorded

| Historical outcome on 150 tasks | Base | Local 14-task | Local 30-task | Harness Hub |
| --- | ---: | ---: | ---: | ---: |
| Verifier pass | 72 | 83 | 100 | 100 |
| Submitted a wrong answer | 31 | 38 | 20 | 15 |
| Missing answer file or empty submission | 47 | 29 | 30 | 35 |

The broader adapter had 17 fewer missing submissions and 11 fewer wrong submissions than base. The extraction joins every released task and arm to one `evaluation/<task>/<arm>/group-0001/trials/*/verifier/test-stdout.txt` under `runs/lfm3-150`, or under `runs/lfm3-breadth` for `broad`. A grader message reporting an absent or empty answer counts as missing. For submissions, the grader's `reward` or `correctness` line distinguishes passing and wrong answers. The checker verifies that each score agrees with the released reward and accounts for all 600 logs. [Extraction and checks](analyze.py) · [Aggregate outcomes](release-audit.json)

## Log-probability replay audit

All **45 optimizer updates** triggered replay at least once. **972 of 2,017 turns (48.2%)** in updated groups exceeded the `0.2`-nat dense-drift threshold and were replayed. The five groups with no update (8, 12, 13, 16, and 43) each had eight zero effective rewards. An interrupted attempt at group 32 had no committed metrics and the replay counts use the completed retry.

| Metric across updated groups | Min | Median | Max |
| --- | ---: | ---: | ---: |
| Replayed turns | 11 | 22 | 34 |
| Dense scorer's maximum drift (nats) | 0.3179 | 0.4492 | 0.7327 |
| Accepted check's maximum drift (nats) | 0.173956 | 0.194153 | 0.199993 |
| Pre-clip gradient norm | 0.1661 | 0.2740 | 0.8117 |
| Maximum adapter parameter change | 3.162 × 10⁻⁵ | 6.158 × 10⁻⁵ | 6.892 × 10⁻⁵ |
| Maximum LoRA B parameter change | 3.162 × 10⁻⁵ | 5.872 × 10⁻⁵ | 6.892 × 10⁻⁵ |

The dense group maximum was between 0.30 and 0.50 nats in 38 updates and above 0.50 in groups 11, 20, 26, 27, 29, 41, and 44. Group 41 had the highest dense drift (0.7327 nats), while group 20 replayed the most turns (34/52). No update reached the global gradient-norm clip threshold of 1. The verifier passed 237/400 training rollouts. Mean effective reward was 0.6315 across all rollouts and 0.7016 in updated groups.

The accepted check uses sampling-shaped scores on replayed turns and dense scores on other turns. The historical trainer retained the dense gradients after successful replay. The held-out evaluation measured the archived adapter with collection only. [Per-group metrics](training.jsonl) · [Scoring methodology](../../docs/methodology.md#log-probability-drift)

### Group 41 gradient audit

The audit used the pre-update group 40 adapter and group 41's 49 retained turns and 7,433 completion tokens. It used the same completion masks, advantages, and per-rollout weighting for all gradient comparisons.

| Group 41 gradients compared | Cosine similarity | Relative L2 difference |
| --- | ---: | ---: |
| Historical dense vs. sampling-shaped cached reference (both MLX 0.32.4) | 0.99687 | 7.93% |
| Historical dense (MLX 0.32.4) vs. patched dense (MLX 0.32.5) | 0.99987 | 1.59% |

The aggregate directions agreed closely for this update. Relative L2 is the norm of the difference divided by the second gradient's norm. Cached forward scores matched the archived sampling scores. The segmented cached backward pass is approximate. On one 126-token turn at block size 24 it differed by 1.12% relative L2 from a direct backward pass. Separately, MLX 0.32.5 failed the historical 0.2-nat replay rule on three turns when it rescored samples collected with 0.32.4. [Gradient comparison](gradient-audit/summary.json) · [Direct backward-pass check](gradient-audit/cached-reference-validation.json) · [Upstream MLX change](https://github.com/ml-explore/mlx/pull/4910)

Effects on other groups and the final trained weights remain open. An extended audit would choose ordinary and high-drift groups in advance and compare at their own collection weights with identical masks, advantages, and per-rollout weighting. [Audit code and per-turn measurements](gradient-audit/README.md)

## Evidence

| File | Contents |
| --- | --- |
| [pairs.jsonl](pairs.jsonl) | Numeric measurements for all four arms with source, seed, and task digest |
| [summary.json](summary.json) | Original broader-arm aggregate with full-precision paired intervals |
| [release-audit.json](release-audit.json) | Input-hash overlap and sensitivity, verifier outcomes, and screening attempts |
| [efficiency-pairs.jsonl](efficiency-pairs.jsonl), [efficiency.json](efficiency.json) | Per-task numeric measurements for the broader and published arms, plus distribution statistics and intervals for all three comparisons |
| [training.jsonl](training.jsonl) | Fifty main groups with verifier scores, rewards, update counts, duration, and memory |
| [provenance.json](provenance.json) | Frozen settings, task order, engine hashes, adapter hashes, and run timing |
| [historical-engine.md](historical-engine.md) | Reconstruct and verify all 18 historical source files |
| [breadth.jsonl](../../examples/smoldataenvs/breadth.jsonl) | Pinned 30-task training schedule and input hashes |

## Regenerate the statistics

Run from the repository root with Python 3.12 or 3.13.

```sh
PYTHONPATH=src python3 results/lfm3-pi-breadth/analyze.py --check
PYTHONPATH=src python3 results/lfm3-pi-breadth/analyze.py --audit-runs runs --check
```

The first command uses the released manifests and rows to repeat the original and sensitivity analyses with 10,000 source-dataset bootstrap draws each. The second also checks the verifier outcome and screening tables against the original archives. [Retraining and evaluation recipe](../../examples/smoldataenvs/README.md) · [Versioned adapter downloads](https://github.com/brendanboyle87/local-apple-harness-adapter/releases/tag/v0.1.0)

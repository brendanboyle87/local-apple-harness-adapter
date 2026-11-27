# LFM3-1.7B BF16 + Pi, 30 training tasks

The broader adapter passed **100/150** held-out SmolDataEnvs tasks. It used 50 groups of eight rollouts, the same starting checkpoint and rollout limits as the 14-task baseline, and a schedule spanning 30 source datasets. Forty-five groups applied optimizer updates. The main phase took about seven hours and peaked at 35.464 GiB of MLX memory.

| Reference | Reference passes | 30-task passes | Paired pass-rate change (95% CI) | Shared successes | Tool-call change on shared successes (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 72 | 100 | +18.67 pp (+10.27 to +26.97) | 65 | −15.72% (−22.83% to −8.09%) |
| 14-task local adapter | 83 | 100 | +11.33 pp (+3.75 to +18.95) | 72 | +7.84% (+0.87% to +15.05%) |
| Published Harness Hub Pi adapter | 100 | 100 | 0.00 pp (−6.29 to +6.62) | 88 | +27.43% (+18.56% to +36.90%) |

The broader arm ran one fresh rollout per task with the same Pi version, verifier, limits, and per-task seed. The three comparison arms reuse the [earlier experiment's measurements](../lfm3-pi/pairs.jsonl). Both the broader adapter and published adapter solved 100 tasks, with 12 unique passes each. The study covers one training seed and a change in the task schedule, including the mix of tasks and the number of optimizer updates.

Across all 150 tasks the 30-task adapter used **20.04% fewer logical tokens** than base and **12.95% more** than the published adapter. It generated **3.51% more completion tokens** than base, with a confidence interval spanning zero. [Efficiency results, timing caveats, and reproduction](efficiency.md)

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

The accepted check uses sampling-shaped scores on replayed turns and dense scores on other turns. The gradients always came from dense scoring. The records quantify score drift at the group level. The specific tokens that triggered replay and the effect on gradients or final adapter weights remain unknown. The held-out evaluation measured the archived adapter with collection only. [Per-group metrics](training.jsonl) · [Scoring methodology](../../docs/methodology.md#log-probability-drift)

## Evidence

| File | Contents |
| --- | --- |
| [pairs.jsonl](pairs.jsonl) | Numeric measurements for all four arms with source, seed, and task digest |
| [summary.json](summary.json) | Original broader-arm aggregate with full-precision paired intervals |
| [efficiency-pairs.jsonl](efficiency-pairs.jsonl), [efficiency.json](efficiency.json) | Per-task numeric measurements for the broader and published arms, plus distribution statistics and intervals for all three comparisons |
| [training.jsonl](training.jsonl) | Fifty main groups with verifier scores, rewards, update counts, duration, and memory |
| [provenance.json](provenance.json) | Frozen settings, task order, engine hashes, adapter hashes, and run timing |
| [breadth.jsonl](../../examples/smoldataenvs/breadth.jsonl) | Pinned 30-task training schedule and input hashes |

## Regenerate the statistics

Run from the repository root with Python 3.12 or 3.13.

```sh
PYTHONPATH=src python3 results/lfm3-pi-breadth/analyze.py --check
PYTHONPATH=src python3 results/lfm3-pi-breadth/analyze.py
```

The checker repeats the original 10,000 source-dataset bootstrap draws. See the [SmolDataEnvs recipe](../../examples/smoldataenvs/README.md) to retrain or evaluate, and the [versioned release](https://github.com/brendanboyle87/local-apple-harness-adapter/releases/tag/v0.1.0) for both MLX Pi adapters.

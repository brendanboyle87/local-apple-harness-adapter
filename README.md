# Local Apple Harness Adapter

Train a LoRA adapter for an agent harness on an Apple Silicon Mac. This project implements Harness Hub's training recipe in native MLX, using real Pi or OpenCode sessions, Harbor tasks with executable verifiers, and a single-update, group-relative policy gradient. The measured single-Mac experiment uses LFM3-1.7B BF16 with Pi on a frozen selection of SmolDataEnvs tasks. [Harness Hub experiment](https://huggingface.co/blog/harness-hub/harness-adapters)

## Results and downloads

The **LFM3-1.7B BF16 + Pi 30-task adapter** passed **100/150** held-out SmolDataEnvs tasks. It inherited 24 warmup rollouts on two tasks, then used 400 main-training rollouts on 30 other tasks. The main phase applied 45 updates in 7.2 hours and peaked at **35.464 GiB in the MLX allocator** on a 128 GB M4 Max. The task-selection archives contain another 376 screening attempts shared across experiments. The frozen manifest lets a reproduction skip screening. [Training budget and evidence](results/lfm3-pi-breadth/README.md#training-budget)

| Checkpoint | Tasks passed | Tool calls vs. base on shared successes |
| --- | ---: | ---: |
| Base LFM3-1.7B BF16 | 72/150 | — |
| [14-task baseline](https://github.com/brendanboyle87/local-apple-harness-adapter/releases/download/v0.1.0/lfm3-1.7b-pi-mlx-14-task-v0.1.0.tar.gz) | 83/150 | 21.8% fewer |
| [30-task adapter (recommended)](https://github.com/brendanboyle87/local-apple-harness-adapter/releases/download/v0.1.0/lfm3-1.7b-pi-mlx-30-task-v0.1.0.tar.gz) | **100/150** | **15.7% fewer** |

The 30-task adapter gained **18.67 percentage points** over base (paired 95% interval **+10.27 to +26.97**). Two identical input files recur in three held-out tasks. Excluding those tasks gives **97/147 versus 72/147**, or **+17.01 points** (paired 95% interval **+8.82 to +25.00**). The intervals resample task sources. The evaluation used one stochastic attempt per task and arm, and the 30-task arm ran after the other arms on the same holdout. Each tool-call comparison uses the tasks both checkpoints solved. Unpack the download and pass its directory to `--adapter PATH` with Pi and the pinned BF16 base. [Sensitivity analysis and verifier outcomes](results/lfm3-pi-breadth/README.md#input-overlap-sensitivity)

The 30-task adapter used **42.4% fewer logical input and output tokens per success** than base (paired 95% CI **−51.2% to −32.9%**), including resources spent on failed attempts. Across all tasks it generated **3.5% more completion tokens** and took **1.2% less elapsed time**, with both intervals spanning zero. The published Harness Hub adapter matched its pass count while using fewer logical tokens. [Efficiency comparisons and limitations](results/lfm3-pi-breadth/efficiency.md)

**Gradient audit.** In the 30-task run, 972 of 2,017 turns in updated groups triggered sampling replay. The trainer retained the dense gradients after successful checks. In group 41, which had the highest dense drift, the aggregate dense and sampling-shaped cached gradients had cosine similarity **0.99687** and a **7.93% relative L2 difference**. Effects on other groups and final weights remain open. [Group 41 audit](results/lfm3-pi-breadth/README.md#group-41-gradient-audit)

[30-task evidence](results/lfm3-pi-breadth/README.md) · [14-task evidence](results/lfm3-pi/README.md) · [Methodology](docs/methodology.md) · [Release and checksums](https://github.com/brendanboyle87/local-apple-harness-adapter/releases/tag/v0.1.0)

## Supported configurations

| Configuration | Evidence |
| --- | --- |
| LFM3-1.7B BF16 + Pi | Evaluated on held-out SmolDataEnvs tasks |
| OpenCode | Implemented harness integration with a Docker-backed tool and trace test |
| Qwen3 | Implemented model backend with local pilot training updates and zero passes in both arms of a 24-task coding pilot |
| Inkling 2 Mini | Experimental model backend with local training rehearsals and a cached backward pass |

## Install

Use an Apple Silicon Mac with [uv](https://docs.astral.sh/uv/) and Docker Desktop running. Run from the repository root.

```sh
uv sync --locked --python 3.12 --extra apple --extra harbor
```

The first run downloads the pinned model and builds the task image. Rollouts run in ARM64 containers with networking disabled. Before loading a model, run `memory_pressure -Q`, check for trainers and model weights held by local inference servers, and leave headroom for macOS and Docker. Run one model workload at a time.

## Quick start

Run a file-writing task with Pi. Inspect `runs/pi-smoke/group-0001/metrics.json` for the verifier score, tool calls, and efficiency measurements. [Evaluation telemetry fields](docs/evaluation-telemetry.md)

```sh
uv run --no-sync apple-harness --backend lfm3 harbor --harness pi \
  --task examples/harbor/write-answer --attempts 1 --collect-only \
  --effort low --max-output 768 --output runs/pi-smoke
```

To try OpenCode, use `--harness opencode --task examples/harbor/write-answer-opencode` in place of the Pi harness and task arguments, add `--max-context 6144`, and use a fresh output directory. For a small sampling and LoRA backward-pass check, run `uv run --no-sync apple-harness --backend lfm3 smoke`.

## Train, evaluate, and resume

Use the [SmolDataEnvs recipe](examples/smoldataenvs/README.md) for a one-group pilot, the full training schedule, and paired evaluation. The recipe pins the model and task manifests and verifies the downloaded inputs.

- **Train.** Pass one or more `--task PATH` arguments and set `--steps N` for the number of rollout groups. Omit `--collect-only` and use multiple attempts per group. Equal effective rewards skip the update. Each Harbor task needs an instruction, a verifier that emits a score from 0 to 1, and a `no-network` ARM64 image or Dockerfile with Pi 1.5.2 or OpenCode 1.18.30. SafeAgents reviews tool calls by default.
- **Evaluate.** Use `--collect-only` for both arms and match held-out tasks, seeds, harness, model, and rollout limits. Add `--adapter PATH` to the adapter arm. The example runner handles pairing, arm order, and the summary.
- **Resume.** After the previous runner exits, repeat the Harbor command with `--resume`, the same output directory, and matching settings. You can increase `--steps`. Keep the group metrics and checkpoint files. Training saves the sampling key and Adam state in `training.safetensors` and saves the adapter after each update. New collection-only runs save the sampling key in `sampling.safetensors`. Older collection checkpoints remain resumable.

## Research context

Harness Hub also studied the recipe at larger scale. [Harness adapters at scale](https://huggingface.co/blog/harness-hub/harness-adapters-at-scale) · [Upstream attribution and implementation details](docs/methodology.md)

Project code is [Apache-2.0](LICENSE). Models and datasets retain their upstream terms. [Third-party notices](THIRD_PARTY_NOTICES)

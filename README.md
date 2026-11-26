# Local Apple Harness Adapter

Train and evaluate LoRA adapters on Apple Silicon with MLX, Harbor verifiers, and a real agent harness. Choose **Pi** or **OpenCode** for rollouts and **LFM3**, **Qwen3**, or **Inkling** for the model.

## Install

Use an Apple Silicon Mac with [uv](https://docs.astral.sh/uv/) and Docker Desktop running.

```sh
uv sync --locked --python 3.12 --extra apple --extra harbor
```

The first run downloads the pinned model and builds the task image. Inference and training run on your Mac. Harbor trials use ARM64 containers with networking disabled. Before loading a model, check memory pressure with `memory_pressure -Q` and check for other model workloads, including weights held by idle servers.

## Try Pi or OpenCode

Run these one at a time from the repo root. Each command gives the harness a file-writing task and records its tools, sampled tokens, and verifier score.

```sh
uv run --no-sync apple-harness --backend lfm3 harbor --harness pi \
  --task examples/harbor/write-answer --attempts 1 --collect-only \
  --effort low --max-output 768 --output runs/pi-smoke

uv run --no-sync apple-harness --backend lfm3 harbor --harness opencode \
  --task examples/harbor/write-answer-opencode --attempts 1 --collect-only \
  --effort low --max-context 6144 --max-output 768 --output runs/opencode-smoke
```

A passing verifier writes `correctness: [1.0]` to the group's `metrics.json`. Run `uv run --no-sync apple-harness --backend lfm3 smoke` for a quick LoRA backward-pass check.

## Train and evaluate

Follow the [SmolDataEnvs recipe](examples/smoldataenvs/README.md) for a seeded training pilot, the longer run, and paired held-out evaluation. It fetches public inputs and checks them against the manifests in this repo.

For your own data, pass `--task PATH` for each local Harbor task and `--steps N` for the number of rollout groups. A task needs an instruction, a `no-network` ARM64 image or Dockerfile with Pi 1.5.2 or OpenCode 1.18.30 installed, and a verifier that emits a score from 0 to 1. The bundled task directories show the layout. SafeAgents reviews tool calls by default.

- **Train.** Omit `--collect-only` and use multiple attempts per group. A group with varied effective rewards can write an adapter. Equal rewards skip the update.
- **Evaluate.** Use `--collect-only` with the same held-out tasks, harness, model, seeds, and limits for both arms. Add `--adapter PATH` to the adapter arm.
- **Resume.** Repeat the training command with `--resume` and the same output directory and settings. You can raise `--steps`. The runner restores adapter weights, Adam state, and the sampling key from the last complete group.

Each group saves `metrics.json`, rollout and verifier traces, and, for an updated group, `adapter/`. Keep the run directory and `training.safetensors` to resume.

## Validation

A local LFM3/Pi run on an M4 Max applied 41 updates across 50 groups at a peak of **35.5 GiB of MLX memory**. Its adapter used **22% fewer tool calls** on shared successes in a 150-task held-out evaluation. The accuracy gain was **+7.3 points**, with a 95% interval spanning zero. See the [recorded results](examples/smoldataenvs/observed.json).

Code is [Apache-2.0](LICENSE). Models and datasets retain their upstream terms.

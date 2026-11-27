# SmolDataEnvs example

Run from the MLX Harness Adapter repository root after the [installation step](../../README.md#install). The manifests pin the public [SmolDataEnvs task revision](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-harbor-train/tree/588744be728257abfb8d4984ec8fab2acbd1f7e1), task order, and input hashes. Preparation downloads the upstream tasks and builds no-network ARM64 images. Training and evaluation use the local model.

## One training group

Stage the first task from the warmup and train with eight rollouts. The reference run applied an update on this task.

```sh
uv run --no-sync python examples/smoldataenvs/prepare.py \
  --split warmup --limit 1 --output runs/smol-pilot-tasks
uv run --no-sync python examples/smoldataenvs/run.py train \
  --tasks runs/smol-pilot-tasks --steps 1 --output runs/smol-pilot
```

Inspect `runs/smol-pilot/group-0001/metrics.json` for `updated` and `effective_rewards`. Add `--harness opencode` to the **prepare** command to try OpenCode with a fresh output directory. The reference results use Pi.

## Full local run

Stage all warmup and training tasks in separate directories. The warmup runs three groups. The main schedule runs 50 groups, eight rollouts each, starting from the warmup weights with fresh Adam state.

```sh
uv run --no-sync python examples/smoldataenvs/prepare.py \
  --split warmup --output runs/smol-warmup-tasks
uv run --no-sync python examples/smoldataenvs/run.py train \
  --tasks runs/smol-warmup-tasks --output runs/smol-warmup

uv run --no-sync python examples/smoldataenvs/prepare.py \
  --split train --output runs/smol-train-tasks
uv run --no-sync python examples/smoldataenvs/run.py train \
  --tasks runs/smol-train-tasks --adapter runs/smol-warmup/group-0003/adapter \
  --output runs/smol-train
```

The commands above reproduce the 14-task baseline schedule. For the recommended checkpoint, prepare the [30-task schedule](breadth.jsonl) with `--split breadth --output runs/smol-breadth-tasks`. Then run the main training command with `--tasks runs/smol-breadth-tasks --output runs/smol-breadth`, using the same warmup adapter. The runner uses 50 groups and seed 1229 for either schedule.

Use the last group containing an `adapter/` directory if a final group skipped its update. The 14-task main run took about eight hours and the 30-task main run took 7.2 hours. Both peaked at about 35.5 GiB in the MLX allocator, with additional headroom needed for macOS and Docker. The frozen schedules let you skip task screening. Repeat a training command with `--resume` to continue at the last committed group. [30-task training budget](../../results/lfm3-pi-breadth/README.md#training-budget)

## Paired evaluation

This evaluates the base and your adapter on the same 150 held-out tasks with matching seeds. Use the same harness as training. For a small check, prepare fewer tasks with `--limit`.

```sh
uv run --no-sync python examples/smoldataenvs/prepare.py \
  --split heldout --output runs/smol-heldout-tasks
uv run --no-sync python examples/smoldataenvs/run.py evaluate \
  --tasks runs/smol-heldout-tasks --adapter runs/smol-train/group-0050/adapter \
  --output runs/smol-eval
```

Point `--adapter` at `runs/smol-breadth/group-0050/adapter` to evaluate the broader run, or unpack either [release download](https://github.com/brendanboyle87/local-apple-harness-adapter/releases/tag/v0.1.0) and pass its adapter directory. Read `summary.json` for pass counts, tool calls on shared successes, and source-dataset bootstrap intervals. Repeating the evaluation command reuses completed measurements. Sampling and environment rebuilds can change individual outcomes. See the [14-task evidence](../../results/lfm3-pi/README.md) and [30-task evidence](../../results/lfm3-pi-breadth/README.md) for recorded measurements.

The `efficiency` section of `summary.json` compares tokens, requests, time, and allocator peaks using the collected attempt measurements. [Telemetry definitions and file paths](../../docs/evaluation-telemetry.md)

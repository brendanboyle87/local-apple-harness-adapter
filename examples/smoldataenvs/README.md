# SmolDataEnvs example

Run from the repository root after the main README's install step. The manifests pin the public [SmolDataEnvs task revision](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-harbor-train/tree/588744be728257abfb8d4984ec8fab2acbd1f7e1), task order, and input hashes. Preparation downloads the upstream tasks and builds no-network ARM64 images. Training and evaluation use the local model.

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

Use the last group containing an `adapter/` directory if a final group skipped its update. The main reference run took about eight hours after preparation and peaked at 35.5 GiB of MLX memory, plus macOS and Docker memory. Repeat a training command with `--resume` to continue at the last committed group.

## Paired evaluation

This evaluates the base and your adapter on the same 150 held-out tasks with matching seeds. Use the same harness as training. For a small check, prepare fewer tasks with `--limit`.

```sh
uv run --no-sync python examples/smoldataenvs/prepare.py \
  --split heldout --output runs/smol-heldout-tasks
uv run --no-sync python examples/smoldataenvs/run.py evaluate \
  --tasks runs/smol-heldout-tasks --adapter runs/smol-train/group-0050/adapter \
  --output runs/smol-eval
```

Read `summary.json` for pass counts, tool calls on shared successes, and source-dataset bootstrap intervals. Repeating the evaluation command reuses completed measurements. The [reference report](observed.json) records the run on one Mac. Sampling and environment rebuilds can change individual outcomes.

# Local Apple Harness Adapter

Train a LoRA adapter online on Apple Silicon using Inkling 2 Mini, native Pi, and Harbor verifiers. Each update uses freshly sampled attempts from the current policy. Run artifacts include the exact prompt and completion token IDs, completion masks, behavior log probabilities, and verifier scores.

This is an early, sequential reproduction of the Hugging Face harness-adapter workflow. It applies rank-8 LoRA to one attention output projection inside Inkling's repeated block. The training objective is clipped token-level GRPO with one optimizer update per group. A successful verifier score receives a bonus of `0.1 × max(0, 1 − tool_calls / 15)` based on tool calls actually made by the agent. [Hugging Face harness adapters](https://huggingface.co/blog/harness-adapters-at-scale) · [Reward recipe](https://huggingface.co/blog/harness-hub/harness-adapters#training-codebase-and-main-hyperparameters)

## Run locally

Use Apple Silicon macOS, Python 3.12 via [uv](https://docs.astral.sh/uv/), and Docker Desktop with Docker Compose. The default checkpoint download requires about 17 GB of disk space. The Python dependencies, model revision, and Pi fixture dependencies are pinned.

```sh
uv sync --locked --python 3.12 --extra apple --extra harbor

# Confirm an actual LoRA backward pass through the looped model.
uv run --locked --extra apple --extra harbor apple-harness smoke

# Collect eight toy attempts, score them, and apply one GRPO update.
uv run --locked --extra apple --extra harbor apple-harness toy-step --output runs/toy

# Run Pi once in a fresh Harbor container and save the trace.
uv run --locked --extra apple --extra harbor apple-harness harbor \
  --task examples/harbor/write-answer --attempts 1 --collect-only \
  --output runs/pi-smoke

# Train on a group of eight attempts with identical task setup.
uv run --locked --extra apple --extra harbor apple-harness harbor \
  --task examples/harbor/write-answer --output runs/pi-train
```

The first Harbor run builds an ARM64-compatible image with Pi 1.5.2. Each attempt runs Pi as an unprivileged container user. Docker disables runtime networking. A loopback relay carries model requests over `docker exec` standard I/O to an in-process OpenEnv capture service. The relay forwards only the chat-completions path and the local rollout credential. Image builds and dependency downloads use the network.

`harbor --task` accepts a reviewed local Harbor task directory with a standalone Linux Dockerfile or prebuilt image, `network_mode = "no-network"`, an unprivileged agent user, and Pi 1.5.2 already installed. The bundled fixture shows the image and verifier layout. Repeat `--task` and use `--steps` to cycle through tasks one group at a time. For base and adapter evaluation, use matching seeds and `--collect-only` on held-out task paths, passing `--adapter PATH` for the adapter run. The adapter path should point to a group's `adapter/` directory, with the same base model, revision and rank used for training.

## SafeAgents

Harbor runs enforce SafeAgents by default (`--safe-agents enforce`). The pinned rules run locally on proposed tool calls before OpenEnv forwards the response to Pi. SafeAgents replaces blocked calls with notices and preserves the sampled token IDs and log probabilities for training. Use `--safe-agents log-only --collect-only` to inspect findings on calls Pi can execute, or `--safe-agents off` to reproduce an unguarded run. [Reference training recipe](https://huggingface.co/blog/harness-hub/harness-adapters#rule-based-checks)

Each attempt file stores the raw verifier `reward`, the `effective_reward`, and the `safe_agents` call verdicts. An enforced block gives the rollout zero training reward, even if the verifier passed. The group metrics report the number of blocks and the effective reward per attempt. Scanner errors invalidate the attempt and stop the group before an optimizer update.

## Read the outputs

Each `group-*` directory contains the OpenEnv attempt records, Harbor trial logs, and `metrics.json`. A group with varying rewards produces an `adapter/` directory if the log-probability and gradient checks pass. Equal rewards skip the update. Each attempt uses a fresh container, with weights fixed until the full group is scored. Run directories and model files are excluded from Git.

To continue an interrupted Harbor run, repeat the original command and add `--resume`. Use the same output path, task content, and options. You can raise `--steps` to extend the target total number of groups. The runner restores the adapter, Adam state, and sampling key at the last complete group. It archives an unfinished group under `interrupted/` and retries it from that checkpoint.

The runner writes `training.safetensors` before committing a group's `metrics.json`. Keep both files and any saved `adapter/` directories when backing up the run. The `--adapter` option starts a new run from saved weights with fresh optimizer state. Groups report separate collection and optimization timing and peak MLX memory.

The trainer compares captured behavior log probabilities to its batched scoring pass. When a turn falls outside `--max-logprob-drift`, it verifies the recorded values by replaying the sampler's cached prefill and decode shapes. Replay runs outside gradient tracking and must meet the same tolerance. The single on-policy update uses detached batched scores as its baseline, the usual one-iteration GRPO case. The two drift measurements are logged separately. Current defaults target short, text-only tasks. Benchmark-scale results and a held-out improvement measurement are still pending. [TRL's one-iteration GRPO baseline](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)

Pass `harbor --gradient-checkpointing` to recompute Inkling decoder activations during the backward pass. This reduces peak optimizer memory at the cost of extra computation. Profile the longest captured turns before raising the rollout context budget.

Pass `harbor --score-block-size 32` to use the sampler's cached prompt prefill in the differentiable scoring pass. This mode checkpoints decoder layers and backpropagates across cache boundaries. It rescans a turn with smaller completion blocks when token log probabilities exceed `--max-logprob-drift`, ending at sampler-shaped one-token decode if needed. The saved group metrics include the accepted block size per turn. Long one-token fallbacks can dominate update time.

## Local validation

On an M4 Max with 128 GB unified memory, `smoke` backpropagated through 43,008 LoRA parameters in the repeated decoder block at a peak of 16.7 GiB of MLX memory. `toy-step` updated those parameters from eight live samples. The bundled Pi fixture also passed its Harbor verifier with a four-turn trace that OpenEnv reconciled against Pi's session record.

The eight-attempt Harbor group captured 1,352 sampled tokens across 23 turns. All eight passed and used one to three tool calls, so the efficiency term provided the reward variation. After the probability verification change, the captured group produced a nonzero LoRA update from the same initial policy. The rollouts took about 145 seconds and the optimizer pass took 171 seconds, with a peak of 71.6 GiB of MLX memory during optimization.

A repeat with SafeAgents enforcement produced a LoRA update from eight passing attempts and 15 allowed tool calls. The Docker test also uses a scripted completion to verify that Pi can make a legitimate workspace write in a response that contains a blocked dummy reward-file write. The raw verifier score is 1, the training reward is 0, and OpenEnv reconciles the original sampled tokens with Pi's trace.

Gradient checkpointing produced a LoRA update from four fresh fixture attempts at 25.3 GiB of peak MLX memory. A backward pass over a recorded 4,020-token Termgrade turn reached 43.3 GiB. Cached sampler replay reproduced the recorded log probabilities for that turn. [Pinned Termgrade dataset](https://huggingface.co/datasets/ai-and/termgrade-environments/tree/f27ea56e45c6f09039f6eaa2e7daa5773a70ccee)

Cached-prefill scoring produced a nonzero gradient on a full 7,733-token Termgrade turn with 0.097 nats maximum drift and 34.7 GiB peak MLX memory. On a 32-token completion prefix with a 5,891-token prompt, LoRA gradient cosine similarity to one-token cached decode exceeded 0.998 for the active matrices before and after loading the pilot adapter.

The base checkpoint and the saved adapter each passed one held-out smoke rollout with one tool call using Harbor's upstream `hello-world-multiple-solutions` instruction and verifier. Its image was replaced with the offline Pi fixture image and the verifier ran in the same container. [Upstream Harbor task](https://github.com/harbor-framework/harbor/tree/6f5d3df26b2324d7ea5b09d4254cc7bc3c8b1fa6/examples/tasks/hello-world-multiple-solutions)

## Development

```sh
uv run --locked --extra apple --extra harbor pytest
uv run --locked --extra apple --extra harbor ruff check src tests
APPLE_HARNESS_DOCKER_TEST=1 uv run --locked --extra apple --extra harbor pytest tests/test_safe_agents.py
```

The project code is Apache-2.0. Model weights and benchmark tasks are obtained separately under their respective upstream licenses. [OpenEnv Harbor integration](https://huggingface.co/docs/openenv/environments/harbor) · [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) · [Pinned MLX checkpoint](https://huggingface.co/mlx-community/Inkling-2-Mini-4bit/tree/9c0e2960609ba7cb40daf92c6f7bf8ed56e1dd62) · [MLX-LM implementation](https://github.com/ml-explore/mlx-lm/pull/2086)

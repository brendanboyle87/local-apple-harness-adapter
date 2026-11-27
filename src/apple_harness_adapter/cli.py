"""Command-line entry point for local harness adaptation."""

import argparse
import asyncio
import json
import platform
import tomllib
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from time import monotonic

from .backends import DEFAULT_MODELS, DEFAULT_REVISIONS, backend_class
from .trace import Rollout

# Used by existing Inkling profiling scripts.
DEFAULT_MODEL = DEFAULT_MODELS["inkling"]
DEFAULT_REVISION = DEFAULT_REVISIONS[DEFAULT_MODEL]


def write_json(path: Path, data) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(data, output, indent=2, allow_nan=False)
        output.write("\n")


def smoke(policy, args) -> dict:
    import mlx.core as mx
    import mlx.nn as nn

    from .grpo import gradient_norm

    prompt = policy.prompt_ids([{"role": "user", "content": "Say hello."}], None, "none")
    turn = policy.sample(prompt, 8, args.temperature)

    def diagnostic_loss(model):
        return -policy.score(model, turn).mean()

    loss, grad = nn.value_and_grad(policy.model, diagnostic_loss)(policy.model)
    norm = gradient_norm(grad)
    drift = mx.max(mx.abs(policy.score(policy.model, turn) - mx.array(turn.old_logprobs))).item()
    result = {
        "target": policy.targets[0],
        "targets": list(policy.targets),
        "trainable_parameters": policy.trainable_count,
        "prompt_tokens": len(prompt),
        "sampled_tokens": len(turn.completion_token_ids),
        "loss": loss.item(),
        "gradient_norm": norm,
        "max_logprob_drift": drift,
        "peak_mlx_gib": round(mx.get_peak_memory() / 2**30, 3),
    }
    if (
        not all(isfinite(result[key]) for key in ("loss", "gradient_norm", "max_logprob_drift"))
        or norm <= 0
    ):
        raise RuntimeError(f"LoRA backward check failed: {result}")
    return result


def toy_step(policy, args) -> dict:
    from .grpo import GRPO

    content = "Pick either red or blue at random. Reply with just the chosen word."
    prompt = policy.toy_prompt_ids([{"role": "user", "content": content}])
    args.output.mkdir(parents=True, exist_ok=False)
    rollouts = []
    answers = []
    for _ in range(args.attempts):
        turn = policy.sample(prompt, 16, args.temperature)
        answer = policy.toy_answer(turn.completion_token_ids)
        rollouts.append(Rollout("pick-red", (turn,), float(answer == "red"), 0))
        answers.append(answer)
    write_json(args.output / "rollouts.json", [asdict(rollout) for rollout in rollouts])
    trainer = GRPO(policy, args.learning_rate, args.max_logprob_drift)
    result = trainer.update(rollouts) | {"answers": answers}
    if result["updated"]:
        policy.save(args.output / "adapter")
    write_json(args.output / "metrics.json", result)
    return result


def harbor_run(policy, args) -> dict:
    import mlx.core as mx
    from harbor.publisher.packager import Packager

    from .checkpoint import CHECKPOINT_VERSION, commit_group, lock_run, prepare_run
    from .grpo import GRPO
    from .harbor import collect_harbor

    trainer = GRPO(
        policy,
        args.learning_rate,
        args.max_logprob_drift,
        gradient_checkpointing=args.gradient_checkpointing,
        score_block_size=args.score_block_size,
    )
    trainer.optimizer.init(policy.model.trainable_parameters())
    config = {key: value for key, value in vars(args).items() if key != "resume"} | {
        "task": [str(path.resolve()) for path in args.task],
        "task_digests": [Packager.compute_content_hash(path)[0] for path in args.task],
        "output": str(args.output.resolve()),
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "backend": policy.backend,
        "harness": getattr(args, "harness", "pi"),
        "model_identity": policy.model_identity,
        "checkpoint_version": CHECKPOINT_VERSION,
    }
    if not args.score_block_size:
        config.pop("score_block_size")
    with lock_run(args.output, args.resume):
        metrics = prepare_run(args.output, config, args.resume, policy, trainer.optimizer)
        for step in range(len(metrics), args.steps):
            task = args.task[step % len(args.task)]
            directory = args.output / f"group-{step + 1:04d}"
            directory.mkdir()
            efficiency = []

            def save_attempt(index, result):
                write_json(directory / f"attempt-{index + 1:02d}.json", result)
                if "efficiency" in result:
                    efficiency.append(
                        {
                            key: value
                            for key, value in result["efficiency"].items()
                            if key != "tool_spans"
                        }
                    )

            start = monotonic()
            mx.reset_peak_memory()
            rollouts = asyncio.run(
                collect_harbor(policy, args, directory / "trials", save_attempt, task)
            )
            collection_seconds = monotonic() - start
            collection_peak_gib = (
                max(
                    [
                        mx.get_peak_memory(),
                        *(row["mlx_group_peak_to_date_bytes"] for row in efficiency),
                    ]
                )
                / 2**30
            )
            update = {"updated": False}
            if not args.collect_only:
                start = monotonic()
                mx.reset_peak_memory()
                update = trainer.update(rollouts) | {
                    "optimization_seconds": round(monotonic() - start, 2),
                    "optimization_peak_mlx_gib": round(mx.get_peak_memory() / 2**30, 3),
                }
            result = update | {
                "task": task.name,
                "harness": config["harness"],
                "correctness": [rollout.correctness for rollout in rollouts],
                "tool_calls": [rollout.tool_calls for rollout in rollouts],
                "safe_agents": args.safe_agents,
                "blocked_tool_calls": [rollout.blocked_tool_calls for rollout in rollouts],
                "effective_rewards": [rollout.reward for rollout in rollouts],
                "turns": [len(rollout.turns) for rollout in rollouts],
                "collection_seconds": round(collection_seconds, 2),
                "collection_peak_mlx_gib": round(collection_peak_gib, 3),
            }
            if efficiency:
                result["efficiency"] = efficiency
            commit_group(directory, result, policy, trainer.optimizer)
            metrics.append(result)
    return {"groups": metrics}


def training_options(parser) -> None:
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-logprob-drift", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="inkling", choices=DEFAULT_MODELS)
    parser.add_argument(
        "--model", help="Hugging Face model ID or local snapshot (defaults to the backend's model)"
    )
    parser.add_argument("--revision", help="Hugging Face checkpoint revision")
    parser.add_argument("--rank", type=int, help="LoRA rank (defaults to the backend's rank)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.8)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("smoke", help="check sampling, scoring and LoRA backward")
    toy = commands.add_parser(
        "toy-step", help="collect and optimize one live verifier-scored group"
    )
    training_options(toy)
    harbor = commands.add_parser("harbor", help="collect and train on local Harbor tasks")
    training_options(harbor)
    harbor.add_argument("--harness", choices=("pi", "opencode"), default="pi")
    harbor.add_argument("--task", type=Path, action="append", required=True)
    harbor.add_argument("--steps", type=int, default=1)
    harbor.add_argument(
        "--resume", action="store_true", help="continue a run in an existing output directory"
    )
    harbor.add_argument("--collect-only", action="store_true")
    harbor.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="recompute decoder activations during the optimizer pass to reduce peak memory",
    )
    harbor.add_argument(
        "--score-block-size",
        type=int,
        default=0,
        help="score with the sampler's cached prefill and up to this many tokens per decode block",
    )
    harbor.add_argument("--adapter", type=Path, help="initialize from a saved adapter")
    harbor.add_argument("--reward-key", default="")
    harbor.add_argument(
        "--safe-agents",
        choices=("enforce", "log-only", "off"),
        default="enforce",
        help="review harness tool calls with SafeAgents (default: enforce)",
    )
    harbor.add_argument("--max-turns", type=int, default=4)
    harbor.add_argument("--max-output", type=int, default=128)
    harbor.add_argument("--max-context", type=int, default=4096)
    harbor.add_argument(
        "--effort", choices=("none", "minimal", "low", "medium", "high"), default="none"
    )
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("Local training requires Apple Silicon macOS")
    if args.command == "harbor":
        if min(args.steps, args.max_turns, args.max_output, args.max_context) < 1:
            parser.error("Steps and token/turn limits must be positive")
        if args.score_block_size < 0:
            parser.error("Score block size must be nonnegative")
        for path in args.task:
            try:
                with (path / "task.toml").open("rb") as task_file:
                    config = tomllib.load(task_file)
            except (OSError, ValueError) as exc:
                parser.error(str(exc))
            if config.get("environment", {}).get("network_mode") != "no-network":
                parser.error(f"{path} must declare [environment] network_mode = 'no-network'")
            if (path / "environment" / "docker-compose.yaml").exists():
                parser.error(f"{path} must use a single Dockerfile or prebuilt image")
    if args.command != "smoke" and args.attempts < (
        1 if getattr(args, "collect_only", False) else 2
    ):
        parser.error("GRPO requires at least two attempts, and collection requires at least one")

    try:
        policy_class = backend_class(args.backend)
        policy_class.validate_scoring(
            getattr(args, "gradient_checkpointing", False), getattr(args, "score_block_size", 0)
        )
    except ModuleNotFoundError as exc:
        if exc.name in {"mlx", "mlx_lm"}:
            parser.error("Install the Apple backend with `uv sync --extra apple`")
        raise
    except ValueError as exc:
        parser.error(str(exc))

    args.model = args.model or DEFAULT_MODELS[args.backend]
    args.revision = args.revision or DEFAULT_REVISIONS.get(args.model)
    if args.rank is None:
        args.rank = policy_class.default_rank
    policy = policy_class(args.model, args.revision, args.rank, args.seed)
    if adapter := getattr(args, "adapter", None):
        policy.load_adapter(adapter)
    result = {"smoke": smoke, "toy-step": toy_step, "harbor": harbor_run}[args.command](
        policy, args
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

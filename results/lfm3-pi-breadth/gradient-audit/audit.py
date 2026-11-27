import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from time import monotonic

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from apple_harness_adapter.backends.lfm3 import LFM3Policy
from apple_harness_adapter.grpo import GRPO, gradient_norm
from apple_harness_adapter.trace import Rollout, Turn, group_advantages


ROOT = Path("runs/lfm3-breadth")
GROUP = ROOT / "training/group-0041"
PRIOR = ROOT / "training/group-0040/adapter"
OUTPUT = Path(__file__).resolve().parent
EXPECTED_ADAPTER = "275663f39ee65986a85d5b952af2d6acfe954d27d7ead124fc5047760eff3e5d"


def read(path):
    return json.loads(path.read_text())


def verify_inputs():
    plan = read(ROOT / "plan.json")
    for name, digest in plan["engine_files_sha256"].items():
        assert hashlib.sha256((ROOT / "engine" / name).read_bytes()).hexdigest() == digest, name
    assert (
        Path(__import__("apple_harness_adapter").__file__).resolve().parent
        == (ROOT / "engine/apple_harness_adapter").resolve()
    )
    with (PRIOR / "adapters.safetensors").open("rb") as file:
        assert hashlib.file_digest(file, "sha256").hexdigest() == EXPECTED_ADAPTER
    return plan


def read_rollouts():
    metrics = read(GROUP / "metrics.json")
    rollouts = []
    for index, path in enumerate(sorted(GROUP.glob("attempt-*.json"))):
        trace = read(path)
        turns = []
        for row in trace["turns"]:
            if row["role"] != "agent" or row["discarded"] or not row["trainable"]:
                continue
            prompt = tuple(row["prompt_token_ids"])
            completion = tuple(row["completion_token_ids"])
            assert row["loss_mask"] == [0] * len(prompt) + [1] * len(completion)
            turns.append(
                Turn(
                    prompt,
                    completion,
                    tuple(row["per_token_logps"]),
                    row["sampling_params"]["temperature"],
                )
            )
        assert len(turns) == metrics["turns"][index]
        rollout = Rollout(
            trace["task_name"],
            tuple(turns),
            metrics["correctness"][index],
            metrics["tool_calls"][index],
            metrics["blocked_tool_calls"][index],
        )
        assert rollout.reward == metrics["effective_rewards"][index]
        rollouts.append(rollout)
    assert len(rollouts) == 8 and sum(len(r.turns) for r in rollouts) == 49
    assert sum(len(t.completion_token_ids) for r in rollouts for t in r.turns) == 7433
    advantages = group_advantages(rollouts)
    assert advantages == metrics["advantages"]
    return rollouts, advantages


def dense_score(policy):
    trainer = GRPO.__new__(GRPO)
    trainer.policy = policy
    evaluate = nn.value_and_grad(policy.model, trainer._turn_loss)

    def score(turn, advantage):
        (loss, raw), gradient = evaluate(policy.model, turn, advantage)
        drift = raw.item()
        accepted = drift
        replayed = False
        if math.isfinite(drift) and drift > 0.2:
            replay = policy.replay_sampling_logprobs(turn)
            accepted = mx.max(mx.abs(replay - mx.array(turn.old_logprobs))).item()
            replayed = True
        return gradient, dict(
            loss=loss.item(),
            raw_drift=drift,
            accepted_drift=accepted,
            replayed=replayed,
            passes_original_rule=math.isfinite(accepted) and accepted <= 0.2,
        )

    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("historical", "patched", "cached"))
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    expected = "0.32.5" if args.mode == "patched" else "0.32.4"
    assert mx.__version__ == expected, (mx.__version__, mx.__file__)
    assert importlib.metadata.version("mlx-lm") == "0.32.3.dev28+ge2b5bbd3b"
    assert (os.getenv("MLX_SDPA_VJP_FALLBACK") == "1") == (args.mode == "cached")
    plan = verify_inputs()
    rollouts, advantages = read_rollouts()
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(52 * 2**30)
    mx.set_cache_limit(8 * 2**30)
    mx.reset_peak_memory()
    print(
        json.dumps(
            {"mode": args.mode, "mlx": mx.__version__, "mlx_path": mx.__file__, "probe": args.probe}
        ),
        flush=True,
    )
    started = monotonic()
    policy = LFM3Policy(plan["model"], plan["revision"], plan["rank"], plan["training_seed"])
    policy.load_adapter(PRIOR)
    mx.eval(policy.model.parameters())
    if args.mode == "cached":
        from cached_reference import cached_score, verify_reference

        verify_reference(policy, rollouts[6].turns[5])
        score = cached_score(policy)
    else:
        score = dense_score(policy)
    if args.probe:
        return
    aggregate = None
    details = []
    for i, (rollout, advantage) in enumerate(zip(rollouts, advantages, strict=True), 1):
        count = sum(len(t.completion_token_ids) for t in rollout.turns)
        weight = 1 / (len(rollouts) * count)
        for j, turn in enumerate(rollout.turns, 1):
            turn_started = monotonic()
            gradient, report = score(turn, advantage)
            gradient = tree_map(lambda t: t * weight, gradient)
            aggregate = (
                gradient if aggregate is None else tree_map(lambda a, b: a + b, aggregate, gradient)
            )
            mx.eval(aggregate)
            row = {
                "attempt": i,
                "turn": j,
                "prompt_tokens": len(turn.prompt_token_ids),
                "completion_tokens": len(turn.completion_token_ids),
                "advantage": advantage,
                "weight": weight,
                "seconds": round(monotonic() - turn_started, 3),
                **report,
            }
            details.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
    tensors = dict(tree_flatten(aggregate))
    assert len(tensors) == 306
    assert all(mx.all(mx.isfinite(tensor)).item() for tensor in tensors.values())
    norm = gradient_norm(aggregate)
    mx.save_safetensors(str(OUTPUT / f"{args.mode}.safetensors"), tensors)
    summary = {
        "mode": args.mode,
        "mlx": mx.__version__,
        "adapter_sha256": EXPECTED_ADAPTER,
        "turns": len(details),
        "completion_tokens": sum(row["completion_tokens"] for row in details),
        "gradient_norm": norm,
        "seconds": round(monotonic() - started, 3),
        "peak_mlx_gib": round(mx.get_peak_memory() / 2**30, 3),
        "details": details,
    }
    if args.mode != "cached":
        summary.update(
            raw_drift=max(r["raw_drift"] for r in details),
            accepted_drift=max(r["accepted_drift"] for r in details),
            replayed_turns=sum(r["replayed"] for r in details),
            failures_of_original_rule=sum(not r["passes_original_rule"] for r in details),
        )
    (OUTPUT / f"{args.mode}.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({k: v for k, v in summary.items() if k != "details"}, allow_nan=False),
        flush=True,
    )


if __name__ == "__main__":
    with Path("runs/.lfm3-workload.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()

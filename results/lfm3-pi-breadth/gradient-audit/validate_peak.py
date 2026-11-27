import fcntl
import json
from math import sqrt
from time import monotonic
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from audit import LFM3Policy, PRIOR, OUTPUT, read_rollouts, verify_inputs
from cached_reference import make_inputs, segmented, transition


def main():
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(52 * 2**30)
    mx.set_cache_limit(8 * 2**30)
    mx.reset_peak_memory()
    assert mx.__version__ == "0.32.4"
    started = monotonic()
    plan = verify_inputs()
    rollouts, advantages = read_rollouts()
    turn = rollouts[2].turns[2]
    advantage = advantages[2]
    assert len(turn.prompt_token_ids) == 2016 and len(turn.completion_token_ids) == 126
    policy = LFM3Policy(plan["model"], plan["revision"], plan["rank"], plan["training_seed"])
    policy.load_adapter(PRIOR)
    inputs, spans = make_inputs(policy, turn, 126)

    def objective(model):
        cache = model.make_cache()
        scores = []
        for span in spans:
            values = transition(model, cache, inputs, span, turn.temperature, False)
            if values is not None:
                scores.append(values)
        selected = mx.concatenate(scores)
        return -advantage * mx.sum(selected), selected

    (_, scores), gradient = nn.value_and_grad(policy.model, objective)(policy.model)
    mx.eval(gradient, scores)
    reference = dict(tree_flatten(gradient))
    norm_squared = sum(mx.sum(x.astype(mx.float32) ** 2).item() for x in reference.values())
    drift = mx.max(mx.abs(scores - mx.array(turn.old_logprobs))).item()
    assert drift <= 0.2
    findings = {}
    print(
        json.dumps(
            {
                "case": "attempt 3 turn 3, direct 126-token backward",
                "archive_drift": drift,
                "wall_seconds": round(monotonic() - started, 3),
            }
        ),
        flush=True,
    )
    for size in (24, 48):
        derivative, replay, trace = segmented(policy, turn, advantage, size)
        segmented_gradient = dict(tree_flatten(derivative))
        assert segmented_gradient.keys() == reference.keys()
        distance = sum(
            mx.sum((segmented_gradient[k] - value).astype(mx.float32) ** 2).item()
            for k, value in reference.items()
        )
        dot = sum(
            mx.sum(segmented_gradient[k].astype(mx.float32) * value.astype(mx.float32)).item()
            for k, value in reference.items()
        )
        new_norm_squared = sum(
            mx.sum(v.astype(mx.float32) ** 2).item() for v in segmented_gradient.values()
        )
        findings[str(size)] = {
            "relative_l2_vs_direct": sqrt(distance / norm_squared),
            "cosine_vs_direct": dot / sqrt(norm_squared * new_norm_squared),
            "forward_drift": replay,
            "recomputation_drift": trace,
        }
        print(json.dumps({size: findings[str(size)]}), flush=True)
    result = {
        "attempt": 3,
        "turn": 3,
        "completion_tokens": 126,
        "direct_archive_drift": drift,
        "block_sizes": findings,
        "peak_mlx_gib": round(mx.get_peak_memory() / 2**30, 3),
        "seconds": round(monotonic() - started, 3),
    }
    (OUTPUT / "cached-reference-validation.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    with Path("runs/.lfm3-workload.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()

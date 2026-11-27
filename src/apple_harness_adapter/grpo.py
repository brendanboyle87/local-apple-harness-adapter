"""Sequential, single-update, group-relative policy gradients."""

from contextlib import nullcontext
from math import isfinite

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from .backends.base import MLXPolicy
from .trace import Rollout, Turn, group_advantages


def gradient_norm(gradients) -> float:
    return mx.sqrt(
        sum(mx.sum(value.astype(mx.float32) ** 2) for _, value in tree_flatten(gradients))
    ).item()


class GRPO:
    def __init__(
        self,
        policy: MLXPolicy,
        learning_rate: float,
        max_logprob_drift: float,
        *,
        gradient_checkpointing: bool = False,
        score_block_size: int = 0,
    ):
        if not isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if not isfinite(max_logprob_drift) or max_logprob_drift <= 0:
            raise ValueError("Log-probability tolerance must be finite and positive")
        if score_block_size < 0:
            raise ValueError("Score block size must be nonnegative")
        policy.validate_scoring(gradient_checkpointing, score_block_size)
        self.policy = policy
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        self.max_logprob_drift = max_logprob_drift
        self.score_block_size = score_block_size
        self.gradient_checkpointing = (
            policy.checkpoint_layers if gradient_checkpointing else nullcontext
        )
        self.loss_and_grad = nn.value_and_grad(policy.model, self._turn_loss)

    def _turn_loss(self, model: nn.Module, turn: Turn, advantage: float):
        logprobs = self.policy.score(model, turn)
        # The weights stay fixed throughout sampling and scoring. With a single
        # update per group, the detached score is the on-policy baseline.
        log_ratio = logprobs - mx.stop_gradient(logprobs)
        ratio = mx.exp(log_ratio)
        clipped = mx.clip(ratio, 0.8, 1.2)
        loss = -mx.minimum(ratio * advantage, clipped * advantage).sum()
        return loss, mx.max(mx.abs(logprobs - mx.array(turn.old_logprobs)))

    def _batched_turn_gradient(self, turn, advantage):
        with self.gradient_checkpointing():
            (loss, drift), grad = self.loss_and_grad(self.policy.model, turn, advantage)
        batched_drift = drift = drift.item()
        replayed = isfinite(drift) and drift > self.max_logprob_drift
        if replayed:
            replay = self.policy.replay_sampling_logprobs(turn)
            drift = mx.max(mx.abs(replay - mx.array(turn.old_logprobs))).item()
        if not isfinite(drift) or drift > self.max_logprob_drift:
            raise RuntimeError(
                f"Behavior and scoring log probabilities differ by {drift:.4f} nats "
                f"(limit {self.max_logprob_drift:.4f})"
            )
        return loss.item(), drift, grad, replayed, batched_drift

    def update(self, rollouts: list[Rollout]) -> dict:
        advantages = group_advantages(rollouts)
        metrics = {
            "rewards": [rollout.reward for rollout in rollouts],
            "advantages": advantages,
            "sampled_tokens": sum(
                len(turn.completion_token_ids) for rollout in rollouts for turn in rollout.turns
            ),
            "updated": False,
        }
        if not any(advantages):
            return metrics | {"reason": "all attempts received the same reward"}

        gradients = None
        total_loss, max_drift, max_batched_drift, replayed_turns = 0.0, 0.0, 0.0, 0
        cached_block_sizes = []
        for rollout, advantage in zip(rollouts, advantages, strict=True):
            count = sum(len(turn.completion_token_ids) for turn in rollout.turns)
            weight = 1 / (len(rollouts) * count)
            for turn in rollout.turns:
                if self.score_block_size:
                    loss, drift, grad, block, raw_drift = self.policy.cached_gradient(
                        turn, advantage, self.score_block_size, self.max_logprob_drift
                    )
                    cached_block_sizes.append(block)
                else:
                    loss, drift, grad, replayed, raw_drift = self._batched_turn_gradient(
                        turn, advantage
                    )
                    replayed_turns += replayed
                max_batched_drift = max(max_batched_drift, raw_drift)
                grad = tree_map(lambda value: value * weight, grad)
                gradients = (
                    grad if gradients is None else tree_map(lambda a, b: a + b, gradients, grad)
                )
                mx.eval(gradients)
                total_loss += loss * weight
                max_drift = max(max_drift, drift)

        norm = gradient_norm(gradients)
        if not isfinite(total_loss) or not isfinite(norm) or norm == 0:
            raise RuntimeError(
                f"Expected a finite loss and nonzero finite gradient, got {total_loss}, {norm}"
            )
        if norm > 1:
            gradients = tree_map(lambda value: value / norm, gradients)
        before = dict(tree_flatten(self.policy.model.trainable_parameters()))
        self.optimizer.update(self.policy.model, gradients)
        mx.eval(self.policy.model.parameters(), self.optimizer.state)
        after = dict(tree_flatten(self.policy.model.trainable_parameters()))
        deltas = {key: mx.max(mx.abs(after[key] - value)).item() for key, value in before.items()}
        delta = max(deltas.values())
        if not all(map(isfinite, deltas.values())) or delta == 0:
            raise RuntimeError("Optimizer did not produce a finite, nonzero LoRA update")
        if cached_block_sizes:
            metrics["cached_decode_blocks"] = cached_block_sizes
        return metrics | {
            "updated": True,
            "loss": total_loss,
            "gradient_norm": norm,
            "max_logprob_drift": max_drift,
            "max_batched_logprob_drift": max_batched_drift,
            "replayed_turns": replayed_turns,
            "max_adapter_delta": delta,
            "max_lora_b_delta": max(deltas[f"{target}.lora_b"] for target in self.policy.targets),
        }

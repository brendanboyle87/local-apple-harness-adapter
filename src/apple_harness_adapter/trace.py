"""Exact per-request policy inputs and verifier rewards."""

from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import fmean


@dataclass(frozen=True)
class Turn:
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    old_logprobs: tuple[float, ...]
    temperature: float

    def __post_init__(self) -> None:
        if not self.prompt_token_ids or not self.completion_token_ids:
            raise ValueError("A turn needs both prompt and sampled tokens")
        if len(self.completion_token_ids) != len(self.old_logprobs):
            raise ValueError("Every sampled token needs its behavior log probability")
        if any(type(token) is not int or token < 0 for token in self.input_ids):
            raise ValueError("Token IDs must be nonnegative integers")
        if any(not isfinite(value) or value > 0 for value in self.old_logprobs):
            raise ValueError("Behavior log probabilities must be finite and nonpositive")
        if not isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Sampling requires a finite, positive temperature")

    @property
    def input_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids + self.completion_token_ids

    @property
    def completion_mask(self) -> tuple[int, ...]:
        return (0,) * len(self.prompt_token_ids) + (1,) * len(self.completion_token_ids)


@dataclass(frozen=True)
class Rollout:
    task: str
    turns: tuple[Turn, ...]
    correctness: float
    tool_calls: int
    blocked_tool_calls: int = 0

    @property
    def reward(self) -> float:
        if not isfinite(self.correctness) or not 0 <= self.correctness <= 1:
            raise ValueError("Verifier reward must be a finite score in [0, 1]")
        if self.tool_calls < 0 or self.blocked_tool_calls < 0:
            raise ValueError("Tool-call counts cannot be negative")
        if self.blocked_tool_calls:
            return 0.0
        efficiency = max(0.0, 1.0 - self.tool_calls / 15)
        return self.correctness + (0.1 * efficiency if self.correctness == 1 else 0)


def group_advantages(rollouts: list[Rollout]) -> list[float]:
    if len(rollouts) < 2 or len({rollout.task for rollout in rollouts}) != 1:
        raise ValueError("GRPO requires at least two attempts at the same task")
    if any(not rollout.turns for rollout in rollouts):
        raise ValueError("Every attempt needs at least one exact turn")
    rewards = [rollout.reward for rollout in rollouts]
    mean = fmean(rewards)
    centered = [reward - mean for reward in rewards]
    std = sqrt(sum(delta**2 for delta in centered) / (len(centered) - 1))
    return [delta / (std + 1e-4) for delta in centered]

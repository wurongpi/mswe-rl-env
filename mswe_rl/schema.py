"""Data types shared by the proxy, the episode runner and trajectory assembly."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LLMCall:
    """One chat-completion call made by the agent, as recorded by the proxy."""

    session_id: str
    index: int
    messages: list[dict[str, Any]]
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    # Per-token logprob of each completion token under the rollout policy.
    # None when the inference engine did not return logprobs.
    completion_logprobs: list[float] | None
    text: str
    finish_reason: str | None = None
    latency_s: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LLMCall":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Segment:
    """A run of LLM calls whose prompts extend each other token-for-token.

    tokens      = prompt of the first call, then for every call: [env delta] + [completion]
    loss_mask   = 1 for tokens the policy sampled, 0 for prompt / tool / env tokens
    logprobs    = rollout logprob for sampled tokens, 0.0 elsewhere
    """

    prompt_length: int
    tokens: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    call_indices: list[int] = field(default_factory=list)
    logprobs_complete: bool = True

    @property
    def response_length(self) -> int:
        return len(self.tokens) - self.prompt_length


@dataclass
class EpisodeResult:
    episode_id: str
    task: str
    # "pass" | "fail" | "env_error"
    status: str
    # 1.0 / 0.0, or None for env_error (must be excluded from training)
    reward: float | None
    agent_exit: str = ""
    num_calls: int = 0
    num_segments: int = 0
    prefix_breaks: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    duration_s: float = 0.0

"""Turn the LLM calls of one episode into token-level training samples.

The key rule: never re-tokenize text. We only use the token ids the inference
engine actually consumed (prompt) and produced (completion).

For call k to extend the same sequence as call k-1, its prompt must start with
  prompt_{k-1} + completion_{k-1}
token-for-token. The tokens between that prefix and the end of prompt_k are
what the environment / agent harness appended (tool output, next user turn),
so they get loss_mask = 0.

When the prefix does not hold (history truncation by the agent, a chat template
that rewrites earlier turns, e.g. stripping <think> blocks), we start a new
segment. Every segment of an episode shares the episode reward.
"""

from __future__ import annotations

from typing import Any

from .schema import EpisodeResult, LLMCall, Segment


def _first_mismatch(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def assemble_segments(calls: list[LLMCall]) -> tuple[list[Segment], list[dict[str, Any]]]:
    """Return (segments, prefix_breaks)."""
    segments: list[Segment] = []
    breaks: list[dict[str, Any]] = []
    cur: Segment | None = None

    for call in sorted(calls, key=lambda c: c.index):
        prompt = list(call.prompt_token_ids)
        comp = list(call.completion_token_ids)
        lps = call.completion_logprobs
        if lps is not None and len(lps) != len(comp):
            # Misaligned logprobs are worse than none: drop them.
            lps = None

        if cur is not None and prompt[: len(cur.tokens)] == cur.tokens:
            delta = prompt[len(cur.tokens) :]
            cur.tokens += delta
            cur.loss_mask += [0] * len(delta)
            cur.logprobs += [0.0] * len(delta)
        else:
            if cur is not None:
                pos = _first_mismatch(prompt, cur.tokens)
                breaks.append(
                    {
                        "at_call": call.index,
                        "mismatch_pos": pos,
                        "prev_len": len(cur.tokens),
                        "new_prompt_len": len(prompt),
                        # truncation usually shows up as a shorter prompt
                        "kind": "shorter_prompt" if len(prompt) < len(cur.tokens) else "rewritten_history",
                    }
                )
                segments.append(cur)
            cur = Segment(
                prompt_length=len(prompt),
                tokens=prompt,
                loss_mask=[0] * len(prompt),
                logprobs=[0.0] * len(prompt),
            )

        cur.tokens += comp
        cur.loss_mask += [1] * len(comp)
        if lps is None:
            cur.logprobs += [0.0] * len(comp)
            cur.logprobs_complete = False
        else:
            cur.logprobs += [float(x) for x in lps]
        cur.call_indices.append(call.index)

    if cur is not None:
        segments.append(cur)
    return segments, breaks


def to_samples(result: EpisodeResult, segments: list[Segment]) -> list[dict[str, Any]]:
    """Framework-neutral samples. One row per segment.

    group_id is the task name: GRPO computes advantages within a group of
    episodes on the same task. All segments of one episode share its reward,
    so they also share its advantage.
    """
    rows = []
    for i, seg in enumerate(segments):
        rows.append(
            {
                "episode_id": result.episode_id,
                "group_id": result.task,
                "segment_index": i,
                "num_segments": len(segments),
                "tokens": seg.tokens,
                "loss_mask": seg.loss_mask,
                "rollout_logprobs": seg.logprobs,
                "logprobs_complete": seg.logprobs_complete,
                "prompt_length": seg.prompt_length,
                "response_length": seg.response_length,
                "reward": result.reward,
                "status": result.status,
            }
        )
    return rows

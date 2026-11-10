"""Agents that can be run inside an episode.

An agent is any callable ``(task_text, sandbox, api_base) -> exit_status``. It talks
to the model only through ``api_base`` (the recording proxy), which is what lets the
environment record token-level data without knowing how the agent works inside.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

from ..sandbox import Sandbox

AgentFn = Callable[[str, Sandbox, str], str]

# First line of the output of the command an agent runs to end the episode.
# Same marker as mini-swe-agent's own environments.
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def mini_swe_agent_fn(model_name: str, **kwargs) -> AgentFn:
    """mini-swe-agent v2 (DefaultAgent, text-based actions)."""
    from .mini_swe import run_mini_swe_agent

    return partial(_run, run_mini_swe_agent, model_name, kwargs)


def simple_agent_fn(model_name: str, **kwargs) -> AgentFn:
    """The ~60-line reference agent; supports history truncation."""
    from .simple import run_simple_agent

    return partial(_run, run_simple_agent, model_name, kwargs)


def _run(runner, model_name, kwargs, task_text, sandbox, api_base) -> str:
    return runner(task_text, sandbox, api_base, model_name, **kwargs)


__all__ = ["AgentFn", "SUBMIT_MARKER", "mini_swe_agent_fn", "simple_agent_fn"]

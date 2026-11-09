"""One episode: sandbox up -> agent runs via proxy -> tests -> token-level samples."""

from __future__ import annotations

import time
import traceback
import uuid
from functools import partial
from typing import Any, Callable

import requests

from .agent_runner import run_mini_swe_agent
from .sandbox import EnvError, make_sandbox
from .schema import EpisodeResult, LLMCall
from .tasks import Task
from .trajectory import assemble_segments, to_samples

# (task_text, sandbox, api_base) -> agent exit status string
AgentFn = Callable[[str, Any, str], str]


def mini_swe_agent_fn(model_name: str, **kw) -> AgentFn:
    return partial(_call_mini, model_name=model_name, kw=kw)


def _call_mini(task_text, sandbox, api_base, *, model_name, kw):
    return run_mini_swe_agent(task_text, sandbox, api_base, model_name, **kw)


def simple_agent_fn(model_name: str, **kw) -> AgentFn:
    from .simple_agent import run_simple_agent

    return partial(_call_simple, run=run_simple_agent, model_name=model_name, kw=kw)


def _call_simple(task_text, sandbox, api_base, *, run, model_name, kw):
    return run(task_text, sandbox, api_base, model_name, **kw)


def fetch_calls(proxy_url: str, sid: str) -> list[LLMCall]:
    r = requests.get(f"{proxy_url}/sessions/{sid}", timeout=30)
    r.raise_for_status()
    calls = [LLMCall.from_dict(c) for c in r.json()["calls"]]
    requests.delete(f"{proxy_url}/sessions/{sid}", timeout=30)
    return calls


def run_episode(task: Task, proxy_url: str, agent_fn: AgentFn, backend: str = "docker",
                sandbox_kwargs: dict | None = None) -> tuple[EpisodeResult, list[dict]]:
    proxy_url = proxy_url.rstrip("/")
    sid = f"{task.name}-{uuid.uuid4().hex[:8]}"
    t0 = time.time()
    res = EpisodeResult(episode_id=sid, task=task.name, status="env_error", reward=None)
    sandbox = make_sandbox(backend, task.image, task.repo_dir, **(sandbox_kwargs or {}))
    try:
        sandbox.start()
        try:
            res.agent_exit = agent_fn(task.instruction, sandbox, f"{proxy_url}/s/{sid}/v1")
        except EnvError:
            raise
        except requests.ConnectionError as e:
            # proxy / inference server down: infrastructure, not the policy
            raise EnvError(f"model endpoint unreachable: {e}") from e
        except Exception as e:
            # Agent-side crash (e.g. unparseable output with a strict harness):
            # the episode still ends and gets graded on the repo state.
            res.agent_exit = f"AgentException: {e!r}"[:300]
        passed, log = sandbox.run_tests(task.tests_dir)
        res.status, res.reward = ("pass", 1.0) if passed else ("fail", 0.0)
    except EnvError as e:
        res.status, res.reward, res.error = "env_error", None, str(e)
    except Exception:
        res.status, res.reward, res.error = "env_error", None, traceback.format_exc()[-800:]
    finally:
        sandbox.stop()

    samples: list[dict] = []
    try:
        calls = fetch_calls(proxy_url, sid)
    except Exception as e:
        calls = []
        res.error = (res.error + f" | fetch_calls: {e}").strip(" |")
    res.num_calls = len(calls)
    if calls and res.reward is not None:
        segments, breaks = assemble_segments(calls)
        res.num_segments, res.prefix_breaks = len(segments), breaks
        samples = to_samples(res, segments)
    res.duration_s = round(time.time() - t0, 2)
    return res, samples

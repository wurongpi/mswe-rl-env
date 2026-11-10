"""One episode: sandbox up -> agent runs via proxy -> tests -> token-level samples."""

from __future__ import annotations

import time
import traceback
import uuid

import requests

from .agents import AgentFn
from .sandbox import EnvError, make_sandbox
from .schema import EpisodeResult, LLMCall
from .tasks import Task
from .trajectory import assemble_segments, to_samples


def fetch_calls(proxy_url: str, sid: str) -> list[LLMCall]:
    r = requests.get(f"{proxy_url}/sessions/{sid}", timeout=30)
    r.raise_for_status()
    calls = [LLMCall.from_dict(c) for c in r.json()["calls"]]
    requests.delete(f"{proxy_url}/sessions/{sid}", timeout=30)
    return calls


def run_episode(
    task: Task, proxy_url: str, agent_fn: AgentFn, backend: str = "docker", sandbox_kwargs: dict | None = None
) -> tuple[EpisodeResult, list[dict]]:
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
            # mini-swe-agent already turns policy mistakes (format errors, step
            # limits) into a normal exit. Anything that escapes is a harness bug
            # or broken setup: never grade it as a policy failure.
            raise EnvError(f"agent crashed: {e!r}"[:300]) from e
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

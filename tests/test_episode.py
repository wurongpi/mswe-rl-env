import shutil
import subprocess

import pytest

from mswe_rl.episode import mini_swe_agent_fn, run_episode, simple_agent_fn


def _mini():
    return mini_swe_agent_fn("scripted", step_limit=10)


def test_mini_swe_agent_pass(proxy_url, tasks):
    res, samples = run_episode(tasks["fix_add"], proxy_url, _mini(), backend="local")
    assert res.status == "pass" and res.reward == 1.0, res
    assert res.agent_exit == "Submitted"
    assert res.num_calls == 3 and res.num_segments == 1 and res.prefix_breaks == []
    s = samples[0]
    assert len(s["tokens"]) == len(s["loss_mask"]) == len(s["rollout_logprobs"])
    # every sampled token carries a logprob, every other token is masked out
    assert all((m == 1) == (lp != 0.0) for m, lp in zip(s["loss_mask"], s["rollout_logprobs"]))
    assert s["group_id"] == "fix_add"


def test_wrong_fix_gets_zero_reward(proxy_url, tasks):
    res, samples = run_episode(tasks["fix_word_count"], proxy_url, _mini(), backend="local")
    assert res.status == "fail" and res.reward == 0.0
    assert samples and samples[0]["reward"] == 0.0


def test_history_truncation_splits_segments(proxy_url, tasks):
    agent = simple_agent_fn("scripted", step_limit=10, max_history=2)
    res, samples = run_episode(tasks["fix_fizzbuzz"], proxy_url, agent, backend="local")
    assert res.status == "pass"
    assert res.prefix_breaks and res.num_segments == len(samples) > 1
    assert {s["reward"] for s in samples} == {1.0}  # segments share the episode reward


def test_unreachable_model_is_env_error(tasks):
    res, samples = run_episode(tasks["fix_add"], "http://127.0.0.1:9", _mini(), backend="local")
    assert res.status == "env_error" and res.reward is None and samples == []


def test_agent_crash_is_env_error_not_zero_reward(proxy_url, tasks):
    def broken_agent(task_text, sandbox, api_base):
        raise ImportError("No module named 'minisweagent'")

    res, samples = run_episode(tasks["fix_add"], proxy_url, broken_agent, backend="local")
    assert res.status == "env_error" and res.reward is None and samples == []
    assert "agent crashed" in res.error


def _docker_ok():
    return shutil.which("docker") and subprocess.run(["docker", "info"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _docker_ok(), reason="no docker daemon (start OrbStack / Colima)")
def test_docker_backend(proxy_url, tasks):
    res, _ = run_episode(tasks["fix_add"], proxy_url, _mini(), backend="docker")
    assert res.status == "pass", res

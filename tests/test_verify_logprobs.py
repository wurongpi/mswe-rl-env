"""The fake model is a bigram model, so its logprobs can be recomputed exactly.
That makes these end-to-end checks of recording + segment assembly + masking."""

import json

import pytest

from mswe_rl.agents import mini_swe_agent_fn, simple_agent_fn
from mswe_rl.episode import run_episode
from mswe_rl.testing.fake_server import BigramScorer
from mswe_rl.verify_logprobs import compare_sample, main, summarize, verify


@pytest.fixture
def samples(proxy_url, tasks):
    _, a = run_episode(
        tasks["fix_add"], proxy_url, mini_swe_agent_fn("scripted", step_limit=10), backend="local"
    )
    # history truncation -> several segments per episode
    agent = simple_agent_fn("scripted", step_limit=10, max_history=2)
    _, b = run_episode(tasks["fix_fizzbuzz"], proxy_url, agent, backend="local")
    assert len(b) > 1
    return a + b


def test_recorded_logprobs_match_a_fresh_forward_pass(samples):
    reports = verify(samples, BigramScorer(), tolerance=1e-9)
    s = summarize(reports)
    assert s["trained_tokens"] > 0 and s["max_abs_diff"] == 0.0 and s["tokens_over_tolerance"] == 0


def test_shifted_mask_is_detected(samples):
    s = dict(samples[0])
    s["loss_mask"] = [0] + s["loss_mask"][:-1]  # mask off by one: points at the wrong tokens
    r = compare_sample(s, BigramScorer().score(s["tokens"]), tolerance=1e-6)
    assert r.over_tolerance > 0


def test_wrong_token_is_detected_and_located(samples):
    s = dict(samples[0])
    pos = max(i for i, m in enumerate(s["loss_mask"]) if m) - 3
    s["tokens"] = list(s["tokens"])
    s["tokens"][pos] = (s["tokens"][pos] + 1) % 256  # as if text had been re-tokenized differently
    r = compare_sample(s, BigramScorer().score(s["tokens"]), tolerance=1e-6)
    assert r.over_tolerance > 0 and r.worst_position in (pos, pos + 1)


def test_length_mismatch_raises(samples):
    s = dict(samples[0])
    with pytest.raises(ValueError):
        compare_sample(s, [0.0], tolerance=0.05)


def test_cli(samples, tmp_path, capsys):
    path = tmp_path / "samples.jsonl"
    path.write_text("\n".join(json.dumps(s) for s in samples) + "\n")
    out = tmp_path / "report.json"
    assert main([str(path), "--engine", "toy", "--json", str(out)]) == 0
    assert "over tolerance   0 tokens" in capsys.readouterr().out
    assert json.loads(out.read_text())["summary"]["samples"] == len(samples)

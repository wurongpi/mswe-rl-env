import requests


def test_proxy_records_tokens_and_strips_them(proxy_url):
    body = {"model": "x", "messages": [{"role": "user", "content": "Please solve: `add(a, b)` is broken"}]}
    r = requests.post(f"{proxy_url}/s/sess1/v1/chat/completions", json=body, timeout=10)
    assert r.status_code == 200
    resp = r.json()
    assert "prompt_token_ids" not in resp and "token_ids" not in resp["choices"][0]
    assert "cat calc.py" in resp["choices"][0]["message"]["content"]

    calls = requests.get(f"{proxy_url}/sessions/sess1", timeout=10).json()["calls"]
    assert len(calls) == 1
    c = calls[0]
    assert c["prompt_token_ids"] and c["completion_token_ids"][-1] == 257  # ends with <|im_end|>
    assert len(c["completion_logprobs"]) == len(c["completion_token_ids"])


def test_sessions_are_isolated(proxy_url):
    body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    requests.post(f"{proxy_url}/s/a/v1/chat/completions", json=body, timeout=10)
    assert requests.get(f"{proxy_url}/sessions/b", timeout=10).json()["calls"] == []

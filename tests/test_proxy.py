import requests

TASK = {"role": "user", "content": "Please solve: `add(a, b)` is broken"}


def _chat(url, sid, messages):
    r = requests.post(
        f"{url}/s/{sid}/v1/chat/completions", json={"model": "x", "messages": messages}, timeout=10
    )
    assert r.status_code == 200, r.text
    return r.json()["choices"][0]["message"]["content"]


def _calls(url, sid):
    return requests.get(f"{url}/sessions/{sid}", timeout=10).json()["calls"]


def test_proxy_answers_like_a_chat_api_and_records_tokens(proxy_url):
    reply = _chat(proxy_url, "s1", [TASK])
    assert "cat calc.py" in reply
    (c,) = _calls(proxy_url, "s1")
    assert c["prompt_token_ids"] and c["completion_token_ids"][-1] == 257  # ends with <|im_end|>
    assert len(c["completion_logprobs"]) == len(c["completion_token_ids"])
    assert c["text"] == reply


def test_later_calls_extend_the_tokens_even_if_the_template_rewrites_history(think_proxy_url):
    messages = [TASK]
    for _ in range(3):
        reply = _chat(think_proxy_url, "s2", messages)
        messages += [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": "<output>ok</output>"},
        ]
    calls = _calls(think_proxy_url, "s2")
    for prev, cur in zip(calls, calls[1:]):
        seen = prev["prompt_token_ids"] + prev["completion_token_ids"]
        assert cur["prompt_token_ids"][: len(seen)] == seen


def test_edited_history_is_rendered_from_scratch(proxy_url):
    reply = _chat(proxy_url, "s3", [TASK])
    _chat(
        proxy_url,
        "s3",
        [TASK, {"role": "assistant", "content": reply + " (edited)"}, {"role": "user", "content": "x"}],
    )
    first, second = _calls(proxy_url, "s3")
    seen = first["prompt_token_ids"] + first["completion_token_ids"]
    assert second["prompt_token_ids"][: len(seen)] != seen


def test_sessions_are_isolated(proxy_url):
    _chat(proxy_url, "a", [{"role": "user", "content": "hi"}])
    assert _calls(proxy_url, "b") == []

"""MLXEngine logic against a stub of mlx-lm that mimics stream_generate's contract:
tokens are yielded inside the loop, EOS (or the max_tokens-th token) only in the
final response together with finish_reason. Runs anywhere; the real engine needs
Apple Silicon (`uv sync --extra mlx`)."""

import sys
import types
from dataclasses import dataclass

import pytest

EOS = 99


class _LP:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v


class _LogprobVec:
    def __getitem__(self, tid):
        return _LP(-0.01 * tid)


@dataclass
class _Resp:
    token: int
    logprobs: object
    finish_reason: str | None = None


class _Tok:
    eos_token_ids = {EOS}

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(t) for t in ids if not (skip_special_tokens and t == EOS))

    def convert_ids_to_tokens(self, t):
        return f"<{t}>"


def _stream_generate(script):
    def gen(model, tok, prompt, max_tokens, sampler):
        assert isinstance(prompt, list)  # token ids, not text
        seq = list(script)
        last = None
        for n, t in enumerate(seq):
            last = t
            if t in tok.eos_token_ids or n + 1 == max_tokens:
                break
            yield _Resp(t, _LogprobVec())
        yield _Resp(last, _LogprobVec(), "stop" if last in tok.eos_token_ids else "length")

    return gen


@pytest.fixture
def mlx_stub(monkeypatch, tmp_path):
    def install(script):
        mod = types.ModuleType("mlx_lm")
        mod.load = lambda path: ("model", _Tok())
        mod.stream_generate = _stream_generate(script)
        su = types.ModuleType("mlx_lm.sample_utils")
        su.make_sampler = lambda temp, top_p: ("sampler", temp, top_p)
        monkeypatch.setitem(sys.modules, "mlx_lm", mod)
        monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", su)
        from mswe_rl.infer_server import MLXEngine

        return MLXEngine(str(tmp_path))  # a local dir: no HF download

    return install


BODY = {"model": "m", "prompt": [1, 2, 3, 1], "max_tokens": 10}


def test_completion_includes_eos_exactly_once(mlx_stub):
    eng = mlx_stub([5, 6, EOS])
    r = eng.complete(BODY)
    c = r["choices"][0]
    assert r["prompt_token_ids"] == [1, 2, 3, 1]
    assert c["token_ids"] == [5, 6, EOS] and c["finish_reason"] == "stop"
    assert c["logprobs"]["token_logprobs"] == [-0.05, -0.06, -0.99]
    assert c["text"] == "5 6"


def test_max_tokens_keeps_last_token_and_reports_length(mlx_stub):
    eng = mlx_stub([5, 6, 7, 8])
    r = eng.complete({**BODY, "max_tokens": 3})
    c = r["choices"][0]
    assert c["token_ids"] == [5, 6, 7] and c["finish_reason"] == "length"


def test_response_is_what_the_proxy_parses(mlx_stub):
    from mswe_rl.proxy import parse_completion

    eng = mlx_stub([5, EOS])
    comp, lps, text, finish = parse_completion(eng.complete(BODY))
    assert comp[-1] == EOS and len(lps) == len(comp) and finish == "stop"


def test_rejects_text_prompts(mlx_stub):
    eng = mlx_stub([5, EOS])
    with pytest.raises(ValueError):
        eng.complete({"prompt": "hello"})

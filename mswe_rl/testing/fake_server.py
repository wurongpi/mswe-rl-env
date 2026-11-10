"""Scripted stand-in for the inference server: same API as mswe-infer, no GPU, no weights.

Used by the test suite and the offline demo. It serves /v1/completions on token ids,
like a real engine behind the recording proxy, and decodes the prompt with the toy
tokenizer only to decide which scripted reply comes next.

Replies come from a YAML script: for each task, a substring of its instruction
and the ordered list of assistant replies. The next reply follows the last
scripted reply found in the prompt, so it stays on script even when the agent
truncates its history.

    python -m mswe_rl.testing.fake_server --script demo/scripts.yaml --port 8000
"""

from __future__ import annotations

import argparse
import hashlib
import time
import uuid

import yaml
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .toy_tokenizer import EMPTY_THINK, IM_END, ToyTokenizer

SUBMIT_REPLY = "Done.\n\n```mswea_bash_command\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"


def bigram_logprob(prev: int, tid: int) -> float:
    """The fake model's log p(tid | prev): a deterministic bigram "model".

    Depending only on the previous token makes it reproducible from the token
    sequence alone, so `BigramScorer` can recompute it exactly. That gives
    `mswe-verify-logprobs` a ground truth to be tested against.
    """
    h = hashlib.md5(f"{prev}:{tid}".encode()).digest()[0]
    return -round(0.01 + h / 255 * 2.0, 4)


class BigramScorer:
    """Scores a token sequence under the fake model (see verify_logprobs.Scorer)."""

    def score(self, tokens: list[int]) -> list[float]:
        return [0.0] + [bigram_logprob(a, b) for a, b in zip(tokens, tokens[1:])]


def _assistant_turns(prompt_text: str) -> list[str]:
    """Contents of the assistant turns already in the prompt (the last part is the open turn)."""
    parts = prompt_text.split("<|im_start|>assistant\n")[1:-1]
    turns = [p.split("<|im_end|>")[0] for p in parts]
    return [t[len(EMPTY_THINK) :] if t.startswith(EMPTY_THINK) else t for t in turns]


def build_app(scripts: list[dict], tok: ToyTokenizer | None = None) -> Starlette:
    tok = tok or ToyTokenizer()

    def next_reply(prompt_text: str) -> str:
        replies = next((s["replies"] for s in scripts if s["match"] in prompt_text), [])
        done = _assistant_turns(prompt_text)
        if not done:
            step = 0
        elif done[-1] in replies:
            step = replies.index(done[-1]) + 1
        else:
            step = len(done)
        return replies[step] if step < len(replies) else SUBMIT_REPLY

    async def completions(request: Request):
        body = await request.json()
        prompt_ids = list(body["prompt"])
        reply = next_reply(tok.decode(prompt_ids))
        comp_ids = tok.encode(reply) + [IM_END]
        seq = prompt_ids[-1:] + comp_ids
        lps = [bigram_logprob(a, b) for a, b in zip(seq, seq[1:])]
        return JSONResponse(
            {
                "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": body.get("model") or "scripted",
                "prompt_token_ids": prompt_ids,
                "choices": [
                    {
                        "index": 0,
                        "text": reply,
                        "token_ids": comp_ids,
                        "logprobs": {"tokens": [str(t) for t in comp_ids], "token_logprobs": lps},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(comp_ids),
                    "total_tokens": len(prompt_ids) + len(comp_ids),
                },
            }
        )

    return Starlette(routes=[Route("/v1/completions", completions, methods=["POST"])])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--script", default="demo/scripts.yaml")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    import uvicorn

    scripts = yaml.safe_load(open(a.script))
    uvicorn.run(build_app(scripts), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()

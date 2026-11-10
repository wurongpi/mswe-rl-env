"""Scripted stand-in for the inference server: same response shape, no GPU, no weights.

Used by the test suite and the offline demo. It tokenizes with a byte-level toy
tokenizer and a ChatML-style template, so prefix consistency behaves like a real
chat model: re-sending the same history yields the same token prefix.

Replies come from a YAML script: for each task, a substring of its instruction
and the ordered list of assistant replies. The next reply follows the last
scripted reply found in the history, so it stays on script even when the agent
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

from .toy_tokenizer import IM_END, ToyTokenizer

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


def build_app(scripts: list[dict], tok: ToyTokenizer | None = None) -> Starlette:
    tok = tok or ToyTokenizer()

    def pick_script(messages) -> list[str]:
        first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
        for s in scripts:
            if s["match"] in first_user:
                return s["replies"]
        return []

    async def chat(request: Request):
        body = await request.json()
        messages = [{"role": m["role"], "content": m.get("content") or ""} for m in body["messages"]]
        replies = pick_script(messages)
        last = next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), None)
        if last is None:
            step = 0
        elif last in replies:
            step = replies.index(last) + 1
        else:
            step = sum(1 for m in messages if m["role"] == "assistant")
        reply = replies[step] if step < len(replies) else SUBMIT_REPLY
        prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True)
        comp_ids = tok.encode(reply) + [IM_END]
        seq = prompt_ids[-1:] + comp_ids
        lps = [bigram_logprob(a, b) for a, b in zip(seq, seq[1:])]
        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "scripted"),
                "prompt_token_ids": prompt_ids,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                        "token_ids": comp_ids,
                        "logprobs": {
                            "content": [{"token": str(t), "logprob": lp} for t, lp in zip(comp_ids, lps)]
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(comp_ids),
                    "total_tokens": len(prompt_ids) + len(comp_ids),
                },
            }
        )

    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])


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

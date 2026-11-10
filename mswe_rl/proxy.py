"""Recording proxy: token-in, token-out between the agent and the inference engine.

The agent speaks the OpenAI chat API to
    http://<proxy>/s/<session_id>/v1/chat/completions
and never knows it is being recorded. For every call the proxy

  1. builds the prompt as TOKEN IDS: on the first call of an episode by rendering the
     messages with the chat template; on later calls by appending only the new
     messages to the tokens the model already consumed and produced
     (see chat_template.py for why re-rendering the whole history is wrong),
  2. sends those ids to the engine's /v1/completions endpoint,
  3. records prompt ids, completion ids and logprobs, and
  4. answers the agent with an ordinary chat completion.

If the agent changed its history (truncation, edits), the new messages do not extend
the previous ones; the proxy then renders them from scratch, and trajectory assembly
records a prefix break.

Engine contract (/v1/completions, the same shape vLLM >= 0.10.2 returns with
"return_token_ids": true):
    request  {"prompt": [int, ...], "max_tokens", "temperature", "top_p", "logprobs": 1, "return_token_ids": true}
    response choices[0].token_ids               -> list[int]   (including the EOS token if generated)
             choices[0].logprobs.token_logprobs -> list[float] (raw distribution, before temperature/top-p)
             choices[0].text, choices[0].finish_reason

Run:
    python -m mswe_rl.proxy --upstream http://127.0.0.1:8000 --tokenizer Qwen/Qwen3-1.7B --port 8100
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .chat_template import ChatTemplate, load_chat_template, normalize_messages
from .schema import LLMCall


def parse_completion(resp: dict[str, Any]) -> tuple[list[int], list[float] | None, str, str | None]:
    """-> (completion token ids, their logprobs or None, text, finish_reason)."""
    choice = (resp.get("choices") or [{}])[0]
    comp_ids = choice.get("token_ids")
    if comp_ids is None:
        raise ValueError(
            "upstream response has no choices[0].token_ids; the engine must support "
            '"return_token_ids" (vLLM >= 0.10.2, or mswe-infer)'
        )
    lps = (choice.get("logprobs") or {}).get("token_logprobs")
    if lps is not None:
        lps = [float(x) for x in lps]
    return list(comp_ids), lps, choice.get("text") or "", choice.get("finish_reason")


@dataclass
class EpisodeState:
    messages: list[dict[str, str]]  # messages of the last call
    tokens: list[int]  # its prompt + completion
    reply: str  # the text we answered with


class Recorder:
    def __init__(self, record_dir: str | None = None):
        self._lock = threading.Lock()
        self._sessions: dict[str, list[LLMCall]] = {}
        self.record_dir = Path(record_dir) if record_dir else None
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)

    def add(self, sid: str, build) -> LLMCall:
        with self._lock:
            calls = self._sessions.setdefault(sid, [])
            call = build(len(calls))
            calls.append(call)
        if self.record_dir:
            with open(self.record_dir / f"{sid}.jsonl", "a") as f:
                f.write(json.dumps(call.to_dict(), ensure_ascii=False) + "\n")
        return call

    def get(self, sid: str) -> list[LLMCall]:
        with self._lock:
            return list(self._sessions.get(sid, []))

    def pop(self, sid: str) -> list[LLMCall]:
        with self._lock:
            return self._sessions.pop(sid, [])


def build_prompt(
    template: ChatTemplate, state: EpisodeState | None, messages: list[dict[str, str]]
) -> list[int]:
    """Extend the episode's tokens if the agent extended its history, else render from scratch."""
    if state is not None:
        n = len(state.messages)
        extends = (
            len(messages) > n + 1
            and messages[:n] == state.messages
            and messages[n] == {"role": "assistant", "content": state.reply}
        )
        if extends:
            return state.tokens + template.continuation(messages[n + 1 :], state.tokens[-1])
    return template.prompt_ids(messages)


def build_app(
    upstream: str,
    template: ChatTemplate,
    inject: dict[str, Any] | None = None,
    record_dir: str | None = None,
    timeout: float = 600.0,
) -> Starlette:
    upstream = upstream.rstrip("/")
    recorder = Recorder(record_dir)
    states: dict[str, EpisodeState] = {}
    states_lock = threading.Lock()
    session = requests.Session()

    async def chat(request: Request):
        sid = request.path_params["sid"]
        body = await request.json()
        if body.get("stream"):
            return JSONResponse({"error": "streaming is not supported by the recorder"}, status_code=400)
        messages = normalize_messages(body.get("messages", []))
        with states_lock:
            state = states.get(sid)
        try:
            prompt = build_prompt(template, state, messages)
        except Exception as e:
            return JSONResponse({"error": f"chat template failed: {e!r}"}, status_code=500)

        req = {
            "model": body.get("model"),
            "prompt": prompt,
            "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens") or 1024,
            "temperature": body.get("temperature", 1.0),
            "top_p": body.get("top_p", 1.0),
            "logprobs": 1,
            "return_token_ids": True,
            "skip_special_tokens": True,
            **(inject or {}),
        }
        t0 = time.time()
        try:
            r = await run_in_threadpool(session.post, f"{upstream}/v1/completions", json=req, timeout=timeout)
        except requests.RequestException as e:
            return JSONResponse({"error": f"upstream unreachable: {e}"}, status_code=502)
        if r.status_code != 200:
            return JSONResponse({"error": r.text}, status_code=r.status_code)
        resp = r.json()
        try:
            comp_ids, lps, text, finish = parse_completion(resp)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        engine_prompt = resp.get("prompt_token_ids")
        if engine_prompt is not None and list(engine_prompt) != prompt:
            return JSONResponse(
                {"error": "engine changed the prompt token ids (e.g. added a BOS token)"}, status_code=500
            )

        with states_lock:
            states[sid] = EpisodeState(messages, prompt + comp_ids, text)
        recorder.add(
            sid,
            lambda idx: LLMCall(
                session_id=sid,
                index=idx,
                messages=messages,
                prompt_token_ids=prompt,
                completion_token_ids=comp_ids,
                completion_logprobs=lps,
                text=text,
                finish_reason=finish,
                latency_s=time.time() - t0,
            ),
        )
        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": finish,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt),
                    "completion_tokens": len(comp_ids),
                    "total_tokens": len(prompt) + len(comp_ids),
                },
            }
        )

    async def get_session(request: Request):
        sid = request.path_params["sid"]
        calls = recorder.get(sid)
        return JSONResponse({"session_id": sid, "calls": [c.to_dict() for c in calls]})

    async def delete_session(request: Request):
        sid = request.path_params["sid"]
        recorder.pop(sid)
        with states_lock:
            states.pop(sid, None)
        return JSONResponse({"ok": True})

    async def health(request: Request):
        return JSONResponse({"ok": True, "upstream": upstream})

    app = Starlette(
        routes=[
            Route("/s/{sid}/v1/chat/completions", chat, methods=["POST"]),
            Route("/s/{sid}/chat/completions", chat, methods=["POST"]),
            Route("/sessions/{sid}", get_session, methods=["GET"]),
            Route("/sessions/{sid}", delete_session, methods=["DELETE"]),
            Route("/health", health, methods=["GET"]),
        ]
    )
    app.state.recorder = recorder
    return app


def main():
    p = argparse.ArgumentParser(description="Token-in, token-out recording proxy")
    p.add_argument("--upstream", default="http://127.0.0.1:8000")
    p.add_argument(
        "--tokenizer",
        required=True,
        help="HF repo id or local directory of the served model's tokenizer (toy for the fake server)",
    )
    p.add_argument("--enable-thinking", action="store_true", help="Qwen3 thinking mode (template variable)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--record-dir", default="runs/calls")
    p.add_argument("--inject", default="{}", help="JSON merged into every upstream request")
    a = p.parse_args()
    import uvicorn

    template = load_chat_template(a.tokenizer, a.enable_thinking)
    uvicorn.run(
        build_app(a.upstream, template, json.loads(a.inject), a.record_dir),
        host=a.host,
        port=a.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()

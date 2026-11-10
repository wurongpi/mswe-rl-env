"""Recording proxy: an OpenAI-compatible pass-through that records token-level data.

The agent (mini-swe-agent via litellm) is pointed at
    http://<proxy>/s/<session_id>/v1
and never knows it is being recorded. The proxy forwards each
/chat/completions request to the real inference engine, extracts
prompt token ids, completion token ids and logprobs from the response,
and stores them per session.

Expected response fields from the engine (our infer_server.py returns these;
recent vLLM returns the same shape when the request sets return_token_ids):
    response["prompt_token_ids"]                       -> list[int]
    response["choices"][0]["token_ids"]                -> list[int]
    response["choices"][0]["logprobs"]["content"][i]["logprob"]

Run:
    python -m mswe_rl.proxy --upstream http://127.0.0.1:8000 --port 8100
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any

import requests
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .schema import LLMCall


def extract_tokens(resp: dict[str, Any]) -> tuple[list[int], list[int], list[float] | None]:
    prompt_ids = resp.get("prompt_token_ids")
    choice = (resp.get("choices") or [{}])[0]
    comp_ids = choice.get("token_ids")
    if prompt_ids is None or comp_ids is None:
        raise ValueError(
            "upstream response has no prompt_token_ids / choices[0].token_ids; "
            "the inference engine must return token ids (see README)"
        )
    lps = None
    content = (choice.get("logprobs") or {}).get("content")
    if content:
        lps = [float(t["logprob"]) for t in content]
    return list(prompt_ids), list(comp_ids), lps


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


def build_app(
    upstream: str,
    inject: dict[str, Any] | None = None,
    record_dir: str | None = None,
    timeout: float = 600.0,
    strip_token_fields: bool = True,
) -> Starlette:
    upstream = upstream.rstrip("/")
    recorder = Recorder(record_dir)
    session = requests.Session()

    async def chat(request: Request):
        sid = request.path_params["sid"]
        body = await request.json()
        if body.get("stream"):
            return JSONResponse({"error": "streaming is not supported by the recorder"}, status_code=400)
        body.update(inject or {})
        t0 = time.time()
        try:
            r = await run_in_threadpool(
                session.post, f"{upstream}/v1/chat/completions", json=body, timeout=timeout
            )
        except requests.RequestException as e:
            return JSONResponse({"error": f"upstream unreachable: {e}"}, status_code=502)
        if r.status_code != 200:
            return JSONResponse({"error": r.text}, status_code=r.status_code)
        resp = r.json()
        try:
            prompt_ids, comp_ids, lps = extract_tokens(resp)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        choice = resp["choices"][0]
        recorder.add(
            sid,
            lambda idx: LLMCall(
                session_id=sid,
                index=idx,
                messages=body.get("messages", []),
                prompt_token_ids=prompt_ids,
                completion_token_ids=comp_ids,
                completion_logprobs=lps,
                text=(choice.get("message") or {}).get("content") or "",
                finish_reason=choice.get("finish_reason"),
                latency_s=time.time() - t0,
            ),
        )
        if strip_token_fields:
            resp.pop("prompt_token_ids", None)
            for c in resp.get("choices", []):
                c.pop("token_ids", None)
        return JSONResponse(resp)

    async def get_session(request: Request):
        sid = request.path_params["sid"]
        calls = recorder.get(sid)
        return JSONResponse({"session_id": sid, "calls": [c.to_dict() for c in calls]})

    async def delete_session(request: Request):
        recorder.pop(request.path_params["sid"])
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
    p = argparse.ArgumentParser(description="Token-recording OpenAI-compatible proxy")
    p.add_argument("--upstream", default="http://127.0.0.1:8000")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--record-dir", default="runs/calls")
    p.add_argument(
        "--inject",
        default="{}",
        help='JSON merged into every request, e.g. \'{"logprobs": true, "return_token_ids": true}\' for vLLM',
    )
    a = p.parse_args()
    import uvicorn

    uvicorn.run(
        build_app(a.upstream, json.loads(a.inject), a.record_dir),
        host=a.host,
        port=a.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()

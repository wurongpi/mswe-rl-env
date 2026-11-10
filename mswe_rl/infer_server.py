"""Minimal token-in, token-out inference server for an M1/M2 Mac (or any CPU/GPU box).

Stands in for vLLM / SGLang during local development. It serves /v1/completions
with the prompt given as token ids (the recording proxy applies the chat template)
and returns what an RL rollout needs, in the same shape as vLLM with
"return_token_ids": true:
  - prompt_token_ids                   (echo of the ids the model consumed)
  - choices[0].token_ids               (the ids it sampled, including the EOS token)
  - choices[0].logprobs.token_logprobs (log p(token) under the model, temperature 1)

Logprobs are taken from the raw logits (before temperature / top-p), which is
what a trainer recomputes at temperature 1. Comparing them with the trainer's
recomputation is how you measure train-inference mismatch.

Models are managed by Hugging Face: weights live in the HF cache
(~/.cache/huggingface, or $HF_HOME), are fetched with `hf download`, and are
loaded by repo id + revision. `--offline` refuses any network access.

Two engines, same response:
    --engine hf    transformers on MPS / CPU / CUDA
    --engine mlx   Apple MLX (mlx-lm); fastest on Apple Silicon, 4/8-bit weights

Run:
    uv run hf download Qwen/Qwen3-1.7B            # once (HF_ENDPOINT=https://hf-mirror.com in mainland China)
    uv run mswe-infer --model Qwen/Qwen3-1.7B --port 8000

    uv run hf download mlx-community/Qwen3-1.7B-4bit
    uv run mswe-infer --engine mlx --model mlx-community/Qwen3-1.7B-4bit
"""

from __future__ import annotations

import argparse
import threading
import time
import uuid
from pathlib import Path

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


def resolve_model(repo_id: str, revision: str | None = None, offline: bool = False) -> str:
    """Return a local snapshot path for a HF repo id (or pass a local dir through)."""
    if Path(repo_id).is_dir():
        return repo_id
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            repo_id,
            revision=revision,
            local_files_only=offline,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
        )
    except Exception as e:
        hint = f"run `uv run hf download {repo_id}` first" if offline else "check network / HF_ENDPOINT"
        raise SystemExit(f"cannot resolve model {repo_id!r} ({e}); {hint}") from e


def pick_device(name: str):
    import torch

    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sampling_params(body: dict) -> tuple[float, float, int]:
    temperature = float(body.get("temperature", 1.0) or 0.0)
    top_p = float(body.get("top_p", 1.0) or 1.0)
    max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or 1024)
    return temperature, top_p, max_new


def prompt_ids_from(body: dict) -> list[int]:
    prompt = body.get("prompt")
    if not (isinstance(prompt, list) and all(isinstance(t, int) for t in prompt)):
        raise ValueError("this server only accepts prompts as token ids (a list of ints)")
    return prompt


def completion_response(
    model: str,
    prompt_ids: list[int],
    comp_ids: list[int],
    logprobs: list[float],
    text: str,
    finish: str,
    token_str,
) -> dict:
    """/v1/completions response, same shape as vLLM's with "return_token_ids": true."""
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "prompt_token_ids": prompt_ids,
        "choices": [
            {
                "index": 0,
                "text": text,
                "token_ids": comp_ids,
                "logprobs": {"tokens": [token_str(t) for t in comp_ids], "token_logprobs": logprobs},
                "finish_reason": finish,
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(comp_ids),
            "total_tokens": len(prompt_ids) + len(comp_ids),
        },
    }


class HFEngine:
    """HF transformers on MPS / CPU / CUDA."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        dtype: str = "auto",
        revision: str | None = None,
        offline: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = pick_device(device)
        if dtype == "auto":
            # fp32 is the safe default for <=2B models on MPS; bf16 halves memory.
            torch_dtype = torch.float32 if self.device in ("cpu", "mps") else torch.bfloat16
        else:
            torch_dtype = getattr(torch, dtype)
        self.model_id = model_id
        self.local_path = resolve_model(model_id, revision, offline)
        self.tok = AutoTokenizer.from_pretrained(self.local_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.local_path, torch_dtype=torch_dtype).to(
            self.device
        )
        self.model.eval()
        self.lock = threading.Lock()  # one generation at a time on a laptop

    def complete(self, body: dict) -> dict:
        torch = self.torch
        prompt_ids = prompt_ids_from(body)
        temperature, top_p, max_new = sampling_params(body)
        gen_kwargs = dict(
            max_new_tokens=max_new,
            do_sample=temperature > 0,
            return_dict_in_generate=True,
            output_logits=True,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs.update(temperature=temperature, top_p=top_p, top_k=0)

        input_ids = torch.tensor([prompt_ids], device=self.device)
        with self.lock, torch.no_grad():
            out = self.model.generate(input_ids, attention_mask=torch.ones_like(input_ids), **gen_kwargs)
        comp_ids = out.sequences[0, len(prompt_ids) :].tolist()
        # logits[i] is the raw distribution that produced comp_ids[i]
        logprobs = [
            torch.log_softmax(out.logits[i][0].float(), dim=-1)[tid].item() for i, tid in enumerate(comp_ids)
        ]
        # keep everything up to and including the first EOS
        eos_ids = self.model.generation_config.eos_token_id
        eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
        finish = "length"
        for i, tid in enumerate(comp_ids):
            if tid in eos_ids:
                comp_ids, logprobs, finish = comp_ids[: i + 1], logprobs[: i + 1], "stop"
                break
        text = self.tok.decode(comp_ids, skip_special_tokens=True)
        return completion_response(
            body.get("model", self.model_id),
            prompt_ids,
            comp_ids,
            logprobs,
            text,
            finish,
            self.tok.convert_ids_to_tokens,
        )


class MLXEngine:
    """Apple MLX via mlx-lm: the fastest option on Apple Silicon, supports quantized weights.

    mlx-lm's stream_generate takes the prompt as token ids and yields, per step,
    the sampled token and the log-probabilities computed before the sampler
    (no temperature), i.e. the same convention as HFEngine. EOS is not yielded
    inside its loop but in the final response (finish_reason "stop"), so
    collecting every yielded token gives the completion with exactly one EOS.
    """

    def __init__(self, model_id: str, revision: str | None = None, offline: bool = False):
        from mlx_lm import load

        self.model_id = model_id
        self.local_path = resolve_model(model_id, revision, offline)
        self.model, self.tok = load(self.local_path)
        self.device = "mlx"
        self.lock = threading.Lock()

    def complete(self, body: dict) -> dict:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt_ids = prompt_ids_from(body)
        temperature, top_p, max_new = sampling_params(body)
        sampler = make_sampler(temp=temperature, top_p=top_p if top_p < 1.0 else 0.0)
        comp_ids: list[int] = []
        logprobs: list[float] = []
        finish = "length"
        with self.lock:
            for r in stream_generate(self.model, self.tok, prompt_ids, max_tokens=max_new, sampler=sampler):
                tid = int(r.token)
                comp_ids.append(tid)
                logprobs.append(float(r.logprobs[tid].item()))
                if r.finish_reason:
                    finish = "stop" if r.finish_reason == "stop" else "length"
        text = self.tok.decode(comp_ids, skip_special_tokens=True)
        return completion_response(
            body.get("model", self.model_id),
            prompt_ids,
            comp_ids,
            logprobs,
            text,
            finish,
            lambda t: self.tok.convert_ids_to_tokens(t),
        )


Engine = HFEngine  # backwards-compatible name


def build_app(engine) -> Starlette:
    async def completions(request: Request):
        body = await request.json()
        try:
            return JSONResponse(await run_in_threadpool(engine.complete, body))
        except Exception as e:  # surface errors to the agent instead of hanging
            return JSONResponse({"error": {"message": repr(e)}}, status_code=500)

    async def models(request: Request):
        return JSONResponse({"object": "list", "data": [{"id": engine.model_id, "object": "model"}]})

    return Starlette(
        routes=[
            Route("/v1/completions", completions, methods=["POST"]),
            Route("/v1/models", models, methods=["GET"]),
        ]
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--engine",
        choices=["hf", "mlx"],
        default="hf",
        help="hf = transformers (MPS/CPU/CUDA); mlx = Apple MLX (use mlx-community/* weights)",
    )
    p.add_argument("--model", default="Qwen/Qwen3-1.7B", help="HF repo id (or a local directory)")
    p.add_argument("--revision", default=None, help="HF branch / tag / commit to pin")
    p.add_argument("--offline", action="store_true", help="only use the local HF cache")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", help="auto | float32 | bfloat16 | float16")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    import uvicorn

    if a.engine == "mlx":
        engine = MLXEngine(a.model, a.revision, a.offline)
    else:
        engine = HFEngine(a.model, a.device, a.dtype, a.revision, a.offline)
    print(f"loaded {a.model} from {engine.local_path} on {engine.device}")
    uvicorn.run(build_app(engine), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()

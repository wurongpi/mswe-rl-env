"""Check recorded rollouts against the model: recompute log-probs and compare.

For every sample, run ONE forward pass over the full token sequence and compare
log p(token_t | tokens_<t) at the trained positions (loss_mask = 1) with the
logprob the inference engine reported during the rollout.

If the two agree, the whole recording chain is right: the token ids are the ones
the model saw, the segments are stitched in the right order, the mask points at
the sampled tokens, and logprobs are aligned with tokens. If they disagree, the
gap is the train-inference mismatch a trainer would silently inherit, e.g.
  - rollout engine and scoring engine differ (MLX 4-bit vs HF bf16),
  - the chat template rewrote history (segments should have been split),
  - logprobs were recorded after temperature / top-p instead of before.

    uv run mswe-verify-logprobs runs/qwen/samples.jsonl --model Qwen/Qwen3-1.7B
    uv run mswe-verify-logprobs runs/qwen/samples.jsonl --engine mlx --model mlx-community/Qwen3-1.7B-4bit
    uv run mswe-verify-logprobs runs/demo/samples.jsonl --engine toy     # fake-server runs

Recorded logprobs are those of the raw distribution (temperature 1, no top-p),
which is also what a forward pass gives, so no sampling parameters are needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from ._term import Style


class Scorer(Protocol):
    def score(self, tokens: list[int]) -> list[float]:
        """Return log p(tokens[t] | tokens[:t]) for every t (index 0 is 0.0)."""


class HFScorer:
    """HF transformers on MPS / CUDA / CPU."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        dtype: str = "auto",
        revision: str | None = None,
        offline: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM

        from .infer_server import pick_device, resolve_model

        self.torch = torch
        self.device = pick_device(device)
        if dtype == "auto":
            torch_dtype = torch.float32 if self.device in ("cpu", "mps") else torch.bfloat16
        else:
            torch_dtype = getattr(torch, dtype)
        path = resolve_model(model_id, revision, offline)
        self.model = (
            AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype).to(self.device).eval()
        )

    def score(self, tokens: list[int]) -> list[float]:
        torch = self.torch
        ids = torch.tensor([tokens], device=self.device)
        with torch.no_grad():
            logits = self.model(ids).logits[0, :-1].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        picked = logprobs.gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1)
        return [0.0] + picked.tolist()


class MLXScorer:
    """Apple MLX via mlx-lm (Apple Silicon)."""

    def __init__(self, model_id: str, revision: str | None = None, offline: bool = False):
        import mlx.core as mx
        from mlx_lm import load

        from .infer_server import resolve_model

        self.mx = mx
        self.model, _ = load(resolve_model(model_id, revision, offline))

    def score(self, tokens: list[int]) -> list[float]:
        mx = self.mx
        logits = self.model(mx.array([tokens]))[0, :-1].astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        picked = mx.take_along_axis(logprobs, mx.array(tokens[1:])[:, None], axis=-1)
        return [0.0] + picked[:, 0].tolist()


@dataclass
class SampleReport:
    episode_id: str
    segment_index: int
    trained_tokens: int
    mean_abs_diff: float
    max_abs_diff: float
    over_tolerance: int
    worst_position: int  # index into tokens of the largest gap (-1 if nothing trained)
    mean_rollout_logprob: float
    mean_recomputed_logprob: float


def compare_sample(sample: dict[str, Any], recomputed: list[float], tolerance: float) -> SampleReport:
    tokens, mask, recorded = sample["tokens"], sample["loss_mask"], sample["rollout_logprobs"]
    if not (len(tokens) == len(mask) == len(recorded) == len(recomputed)):
        raise ValueError(
            f"length mismatch in sample {sample.get('episode_id')}: tokens={len(tokens)} "
            f"mask={len(mask)} rollout_logprobs={len(recorded)} recomputed={len(recomputed)}"
        )
    idx = [i for i, m in enumerate(mask) if m]
    diffs = [abs(recorded[i] - recomputed[i]) for i in idx]
    n = len(idx)
    worst = max(range(n), key=diffs.__getitem__) if n else -1
    return SampleReport(
        episode_id=sample.get("episode_id", ""),
        segment_index=sample.get("segment_index", 0),
        trained_tokens=n,
        mean_abs_diff=sum(diffs) / n if n else 0.0,
        max_abs_diff=max(diffs) if n else 0.0,
        over_tolerance=sum(d > tolerance for d in diffs),
        worst_position=idx[worst] if n else -1,
        mean_rollout_logprob=sum(recorded[i] for i in idx) / n if n else 0.0,
        mean_recomputed_logprob=sum(recomputed[i] for i in idx) / n if n else 0.0,
    )


def verify(samples: list[dict[str, Any]], scorer: Scorer, tolerance: float = 0.05) -> list[SampleReport]:
    return [compare_sample(s, scorer.score(s["tokens"]), tolerance) for s in samples]


def summarize(reports: list[SampleReport]) -> dict[str, Any]:
    total = sum(r.trained_tokens for r in reports)
    return {
        "samples": len(reports),
        "trained_tokens": total,
        "mean_abs_diff": sum(r.mean_abs_diff * r.trained_tokens for r in reports) / total if total else 0.0,
        "max_abs_diff": max((r.max_abs_diff for r in reports), default=0.0),
        "tokens_over_tolerance": sum(r.over_tolerance for r in reports),
        "samples_over_tolerance": sum(r.over_tolerance > 0 for r in reports),
    }


def load_samples(path: Path, limit: int | None = None) -> tuple[list[dict[str, Any]], int]:
    """Return (usable samples, number skipped because the engine returned no logprobs)."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    usable = [r for r in rows if r.get("logprobs_complete", True)]
    skipped = len(rows) - len(usable)
    return (usable[:limit] if limit else usable), skipped


def make_scorer(engine: str, model: str | None, **kw) -> Scorer:
    if engine == "toy":
        from .testing.fake_server import BigramScorer

        return BigramScorer()
    if not model:
        raise SystemExit("--model is required for the hf and mlx engines")
    if engine == "mlx":
        return MLXScorer(model, kw.get("revision"), kw.get("offline", False))
    return HFScorer(
        model, kw.get("device", "auto"), kw.get("dtype", "auto"), kw.get("revision"), kw.get("offline", False)
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Recompute log-probs with the model and compare to the rollout")
    p.add_argument("samples", help="samples.jsonl written by mswe-rollout")
    p.add_argument("--engine", choices=["hf", "mlx", "toy"], default="hf")
    p.add_argument("--model", help="HF repo id or local directory")
    p.add_argument("--revision", default=None)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument(
        "--tolerance", type=float, default=0.05, help="max |Δ logprob| per token counted as a match"
    )
    p.add_argument("--limit", type=int, default=None, help="only check the first N samples")
    p.add_argument("--json", dest="json_out", default=None, help="write the full report to this file")
    p.add_argument("--color", action="store_true")
    a = p.parse_args(argv)
    style = Style(a.color)

    samples, skipped = load_samples(Path(a.samples), a.limit)
    if not samples:
        print("no samples with complete logprobs to verify")
        return 1
    scorer = make_scorer(
        a.engine, a.model, revision=a.revision, offline=a.offline, device=a.device, dtype=a.dtype
    )
    reports = verify(samples, scorer, a.tolerance)
    summary = summarize(reports)

    for r in reports:
        ok = r.over_tolerance == 0
        tag = style("green", "match   ") if ok else style("red", "MISMATCH")
        where = "" if ok else f"  worst at token {r.worst_position}, {r.over_tolerance} over tolerance"
        print(
            f"  {tag} {r.episode_id} seg {r.segment_index}  trained={r.trained_tokens:<5} "
            f"mean|Δ|={r.mean_abs_diff:.4f}  max|Δ|={r.max_abs_diff:.4f}{where}"
        )
    print(style("bold", "\n── logprob check ─────────────────────────"))
    print(
        f"  samples          {summary['samples']}" + (f" ({skipped} skipped: no logprobs)" if skipped else "")
    )
    print(f"  trained tokens   {summary['trained_tokens']}")
    print(f"  mean |Δ logprob| {summary['mean_abs_diff']:.4f}")
    print(f"  max  |Δ logprob| {summary['max_abs_diff']:.4f}")
    print(
        f"  over tolerance   {summary['tokens_over_tolerance']} tokens in "
        f"{summary['samples_over_tolerance']} samples (tolerance {a.tolerance})"
    )
    if a.json_out:
        Path(a.json_out).write_text(
            json.dumps({"summary": summary, "samples": [asdict(r) for r in reports]}, indent=2)
        )
    return 0 if summary["tokens_over_tolerance"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

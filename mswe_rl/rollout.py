"""Batch rollout: G episodes per task (a GRPO group), N in parallel.

    uv run mswe-rollout --tasks tasks --agent mini --model Qwen/Qwen3-1.7B \
        --backend docker --group-size 4 --concurrency 2 --out runs/samples.jsonl

Writes:
  <out>                 one training sample per line (see trajectory.to_samples)
  <out>.episodes.jsonl  one EpisodeResult per line (status, reward, prefix breaks, ...)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from .episode import mini_swe_agent_fn, run_episode, simple_agent_fn
from .tasks import load_tasks

C = {"pass": "\033[32m", "fail": "\033[31m", "env_error": "\033[33m", "dim": "\033[2m", "bold": "\033[1m",
     "end": "\033[0m"}


def _c(key: str, text: str) -> str:
    return f"{C[key]}{text}{C['end']}" if sys.stdout.isatty() or _FORCE_COLOR else text


_FORCE_COLOR = False


def main(argv=None):
    global _FORCE_COLOR
    p = argparse.ArgumentParser(description="Collect token-level RL rollouts with mini-swe-agent")
    p.add_argument("--tasks", default="tasks")
    p.add_argument("--only", nargs="*", help="task names to run")
    p.add_argument("--proxy", default="http://127.0.0.1:8100")
    p.add_argument("--agent", choices=["mini", "simple"], default="mini")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--model-backend", choices=["proxy", "litellm"], default="proxy",
                   help="mini only: plain HTTP model or mini-swe-agent's LitellmTextbasedModel")
    p.add_argument("--backend", choices=["docker", "local"], default="docker")
    p.add_argument("--group-size", type=int, default=2, help="episodes per task (GRPO group)")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--step-limit", type=int, default=15)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-history", type=int, default=None, help="simple agent only: keep last N messages")
    p.add_argument("--out", default="runs/samples.jsonl")
    p.add_argument("--color", action="store_true", help="force ANSI colors")
    a = p.parse_args(argv)
    _FORCE_COLOR = a.color

    tasks = load_tasks(Path(a.tasks), a.only)
    if not tasks:
        sys.exit(f"no tasks found in {a.tasks}")
    common = dict(step_limit=a.step_limit, temperature=a.temperature, max_tokens=a.max_tokens)
    if a.agent == "mini":
        agent_fn = mini_swe_agent_fn(a.model, model_backend=a.model_backend, **common)
    else:
        agent_fn = simple_agent_fn(a.model, max_history=a.max_history, **common)

    jobs = [t for t in tasks for _ in range(a.group_size)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ep_path = out.with_suffix(out.suffix + ".episodes.jsonl")
    print(_c("bold", f"▶ {len(jobs)} episodes = {len(tasks)} tasks × group {a.group_size}, "
                     f"agent={a.agent}, backend={a.backend}, concurrency={a.concurrency}"))

    t0 = time.time()
    results, n_samples = [], 0
    with open(out, "w") as fs, open(ep_path, "w") as fe, ThreadPoolExecutor(a.concurrency) as pool:
        futs = [pool.submit(run_episode, t, a.proxy, agent_fn, a.backend) for t in jobs]
        for f in as_completed(futs):
            res, samples = f.result()
            results.append(res)
            for s in samples:
                fs.write(json.dumps(s) + "\n")
            n_samples += len(samples)
            fe.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")
            reward = "-" if res.reward is None else f"{res.reward:.0f}"
            tag = _c(res.status, f"{res.status:<9}")
            extra = f"  {_c('dim', res.error[:80])}" if res.error else ""
            print(f"  {tag} {res.task:<16} reward={reward}  calls={res.num_calls:<2} "
                  f"segments={res.num_segments} breaks={len(res.prefix_breaks)} "
                  f"exit={res.agent_exit:<14} {res.duration_s:5.1f}s{extra}")

    dt = time.time() - t0
    n = len(results)
    cnt = {k: sum(r.status == k for r in results) for k in ("pass", "fail", "env_error")}
    valid = n - cnt["env_error"]
    print(_c("bold", "\n── summary ──────────────────────────────"))
    print(f"  pass rate        {cnt['pass']}/{valid} valid episodes"
          + (f" ({cnt['pass'] / valid:.0%})" if valid else ""))
    print(f"  env errors       {cnt['env_error']} (excluded from training)")
    print(f"  prefix breaks    {sum(len(r.prefix_breaks) for r in results)} "
          f"across {sum(1 for r in results if r.prefix_breaks)} episodes")
    print(f"  samples written  {n_samples} → {out}")
    print(f"  throughput       {n / dt * 3600:.0f} episodes/hour ({dt:.1f}s total)")


if __name__ == "__main__":
    main()

"""Summarize a rollout: is this task set usable as an RL environment for this model?

    uv run mswe-report runs/qwen/samples.jsonl
    uv run mswe-report runs/qwen/samples.jsonl --tasks tasks/humanevalfix --json runs/qwen/report.json

Reads <samples>.episodes.jsonl (one row per episode) and the samples file itself.

The number that matters most for GRPO is "mixed" tasks: tasks whose group has both
passing and failing episodes. A group that is all-pass or all-fail has zero
advantage for every sample and teaches the policy nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from ._term import Style
from .tasks import load_tasks


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def build_report(
    episodes: list[dict[str, Any]], samples: list[dict[str, Any]], bug_types: dict[str, str] | None = None
) -> dict[str, Any]:
    valid = [e for e in episodes if e["status"] != "env_error"]
    by_task: dict[str, list[float]] = defaultdict(list)
    for e in valid:
        by_task[e["task"]].append(e["reward"])
    groups = {t: r for t, r in by_task.items() if len(r) >= 2}
    mixed = [t for t, r in groups.items() if 0 < sum(r) < len(r)]
    calls = sum(e["num_calls"] for e in valid)
    report: dict[str, Any] = {
        "episodes": len(episodes),
        "valid_episodes": len(valid),
        "env_error_rate": 1 - len(valid) / len(episodes) if episodes else 0.0,
        "pass_rate": _mean([e["reward"] for e in valid]),
        "tasks": len(by_task),
        "tasks_solved_at_least_once": sum(1 for r in by_task.values() if sum(r) > 0),
        "groups": len(groups),
        "groups_all_pass": sum(1 for r in groups.values() if sum(r) == len(r)),
        "groups_all_fail": sum(1 for r in groups.values() if sum(r) == 0),
        "groups_mixed": len(mixed),
        "mean_calls_per_episode": calls / len(valid) if valid else 0.0,
        "episodes_with_prefix_break": sum(1 for e in valid if e["prefix_breaks"]),
        "prefix_breaks_per_call": sum(len(e["prefix_breaks"]) for e in valid) / calls if calls else 0.0,
        "mean_episode_seconds": _mean([e["duration_s"] for e in valid]),
        "samples": len(samples),
        "mean_tokens_per_sample": _mean([len(s["tokens"]) for s in samples]),
        "trained_token_fraction": (
            sum(sum(s["loss_mask"]) for s in samples) / max(1, sum(len(s["tokens"]) for s in samples))
        ),
        "samples_missing_logprobs": sum(1 for s in samples if not s.get("logprobs_complete", True)),
    }
    if bug_types and any(bug_types.values()):
        per: dict[str, list[float]] = defaultdict(list)
        for e in valid:
            per[bug_types.get(e["task"]) or "unknown"].append(e["reward"])
        report["pass_rate_by_bug_type"] = {
            k: {"episodes": len(v), "pass_rate": _mean(v)} for k, v in sorted(per.items())
        }
    return report


def pct(x: float) -> str:
    return f"{x:.1%}"


def print_report(r: dict[str, Any], style: Style) -> None:
    print(style("bold", "── rollout report ────────────────────────"))
    print(
        f"  episodes             {r['episodes']} ({r['valid_episodes']} valid, env errors {pct(r['env_error_rate'])})"
    )
    print(f"  pass rate            {pct(r['pass_rate'])} of valid episodes")
    print(f"  tasks solved ≥ once  {r['tasks_solved_at_least_once']}/{r['tasks']}")
    if r["groups"]:
        print(
            f"  reward signal        {r['groups_mixed']}/{r['groups']} groups mixed "
            f"({r['groups_all_pass']} all-pass, {r['groups_all_fail']} all-fail)"
        )
    else:
        print("  reward signal        n/a (needs --group-size ≥ 2)")
    print(f"  calls per episode    {r['mean_calls_per_episode']:.1f}")
    print(
        f"  prefix breaks        {r['episodes_with_prefix_break']}/{r['valid_episodes']} episodes, "
        f"{pct(r['prefix_breaks_per_call'])} of calls"
    )
    print(
        f"  samples              {r['samples']}, {r['mean_tokens_per_sample']:.0f} tokens each, "
        f"{pct(r['trained_token_fraction'])} trained"
    )
    if r["samples_missing_logprobs"]:
        print(style("yellow", f"  missing logprobs     {r['samples_missing_logprobs']} samples"))
    print(f"  episode duration     {r['mean_episode_seconds']:.1f}s mean")
    for k, v in r.get("pass_rate_by_bug_type", {}).items():
        print(f"    {k:<18} {pct(v['pass_rate']):>6}  ({v['episodes']} episodes)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Summarize a rollout (pass rate, reward signal, prefix breaks)")
    p.add_argument("samples", help="samples.jsonl written by mswe-rollout")
    p.add_argument("--tasks", help="task set directory, to break results down by metadata.bug_type")
    p.add_argument("--json", dest="json_out", help="also write the report to this file")
    p.add_argument("--color", action="store_true")
    a = p.parse_args(argv)

    samples_path = Path(a.samples)
    episodes_path = samples_path.with_suffix(samples_path.suffix + ".episodes.jsonl")
    if not episodes_path.exists():
        print(f"missing {episodes_path} (written next to the samples by mswe-rollout)")
        return 1
    bug_types = None
    if a.tasks:
        bug_types = {t.name: t.metadata.get("bug_type", "") for t in load_tasks(Path(a.tasks))}
    report = build_report(_read_jsonl(episodes_path), _read_jsonl(samples_path), bug_types)
    print_report(report, Style(a.color))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

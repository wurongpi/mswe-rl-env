"""Check that a task set grades correctly, without any model.

For every task:
  - the untouched repo must FAIL the hidden tests (otherwise reward 1 is free), and
  - the reference solution (task/solution/) must PASS them (otherwise reward 1 is impossible).

    uv run mswe-check-tasks tasks/toy
    uv run mswe-check-tasks tasks/humanevalfix --backend docker --concurrency 4
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from ._term import Style
from .sandbox import EnvError, make_sandbox
from .tasks import Task, load_tasks


@dataclass
class TaskCheck:
    task: str
    buggy_fails: bool | None = None  # None = could not run
    solution_passes: bool | None = None  # None = no solution/ or could not run
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.buggy_fails is True and self.solution_passes is True


def _graded(task: Task, backend: str, apply_solution: bool) -> bool:
    sandbox = make_sandbox(backend, task.image, task.repo_dir)
    try:
        sandbox.start()
        if apply_solution:
            assert task.solution_dir is not None
            sandbox.write_files(task.solution_dir)
        passed, _ = sandbox.run_tests(task.tests_dir)
        return passed
    finally:
        sandbox.stop()


def check_task(task: Task, backend: str = "local") -> TaskCheck:
    res = TaskCheck(task.name)
    try:
        res.buggy_fails = not _graded(task, backend, apply_solution=False)
        if task.solution_dir is None:
            res.error = "no solution/ directory"
        else:
            res.solution_passes = _graded(task, backend, apply_solution=True)
    except EnvError as e:
        res.error = f"env error: {e}"
    return res


def check_tasks(tasks: list[Task], backend: str = "local", concurrency: int = 4) -> list[TaskCheck]:
    with ThreadPoolExecutor(concurrency) as pool:
        return list(pool.map(lambda t: check_task(t, backend), tasks))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Verify that buggy repos fail and reference solutions pass")
    p.add_argument("tasks", help="task set directory")
    p.add_argument("--backend", choices=["local", "docker"], default="local")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--color", action="store_true")
    a = p.parse_args(argv)
    style = Style(a.color)

    results = check_tasks(load_tasks(Path(a.tasks)), a.backend, a.concurrency)
    bad = [r for r in results if not r.ok]
    for r in bad:
        why = r.error or (
            "untouched repo passes the tests"
            if r.buggy_fails is False
            else "reference solution fails the tests"
        )
        print(f"  {style('red', 'BAD')}  {r.task:<28} {why}")
    n = len(results)
    print(
        f"{n - len(bad)}/{n} tasks grade correctly "
        f"(untouched repo fails, reference solution passes) on the {a.backend} backend"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

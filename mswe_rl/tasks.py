"""Task format (one directory per task):

<task_set>/<name>/
  task.yaml       instruction: str
                  image: str            (default python:3.11-slim)
                  metadata: mapping     (optional, e.g. bug_type; copied into reports)
  repo/           files copied into the sandbox workspace
  tests/test.sh   run after the agent finishes; exit 0 = pass.
                  Gets $WORKSPACE and $TESTS_DIR in its environment.
  solution/       optional reference solution: files copied over the workspace.
                  Never shown to the agent; used by `mswe-check-tasks`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_IMAGE = "python:3.11-slim"


@dataclass
class Task:
    name: str
    instruction: str
    image: str
    repo_dir: Path
    tests_dir: Path
    solution_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_task(path: Path) -> Task:
    path = Path(path)
    meta = yaml.safe_load((path / "task.yaml").read_text())
    solution = path / "solution"
    return Task(
        name=path.name,
        instruction=meta["instruction"].strip(),
        image=meta.get("image", DEFAULT_IMAGE),
        repo_dir=path / "repo",
        tests_dir=path / "tests",
        solution_dir=solution if solution.is_dir() else None,
        metadata=meta.get("metadata") or {},
    )


def load_tasks(root: Path, names: list[str] | None = None) -> list[Task]:
    """Load every task directory directly under ``root`` (optionally only ``names``)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"task set not found: {root}")
    dirs = sorted(p for p in root.iterdir() if (p / "task.yaml").exists())
    if names:
        dirs = [d for d in dirs if d.name in names]
    return [load_task(d) for d in dirs]

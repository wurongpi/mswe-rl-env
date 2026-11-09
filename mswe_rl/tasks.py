"""Task format (one directory per task):

    tasks/<name>/
      task.yaml       instruction: str, image: str (default python:3.11-slim)
      repo/           files copied into the sandbox workspace
      tests/test.sh   run after the agent finishes; exit 0 = pass.
                      Gets $WORKSPACE and $TESTS_DIR in its environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Task:
    name: str
    instruction: str
    image: str
    repo_dir: Path
    tests_dir: Path


def load_task(path: Path) -> Task:
    path = Path(path)
    meta = yaml.safe_load((path / "task.yaml").read_text())
    return Task(
        name=path.name,
        instruction=meta["instruction"].strip(),
        image=meta.get("image", "python:3.11-slim"),
        repo_dir=path / "repo",
        tests_dir=path / "tests",
    )


def load_tasks(root: Path, names: list[str] | None = None) -> list[Task]:
    dirs = sorted(p for p in Path(root).iterdir() if (p / "task.yaml").exists())
    if names:
        dirs = [d for d in dirs if d.name in names]
    return [load_task(d) for d in dirs]

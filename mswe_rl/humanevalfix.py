"""Convert HumanEvalFix (the bug-fixing split of HumanEvalPack, Python) into tasks.

Source: https://huggingface.co/datasets/bigcode/humanevalpack (MIT). 164 problems;
each has a function with a human-written bug, the correct solution and unit tests.

    uv run mswe-make-humanevalfix --out tasks/humanevalfix            # downloads via the HF hub
    uv run mswe-make-humanevalfix --input humanevalpack.jsonl --out tasks/humanevalfix
    uv run mswe-check-tasks tasks/humanevalfix                        # then verify the grading

Each task becomes:
    repo/solution.py        imports + signature (+ docstring) + buggy body
    solution/solution.py    same with the correct body (never shown to the agent)
    tests/test_solution.py  the dataset's `check` function, with a 30 s alarm
    tests/test.sh
    task.yaml               instruction + metadata (bug_type, failure_symptoms)

--hint docstring (default) keeps the docstring in the file, so the intended
behaviour is specified. --hint none drops it: the agent only sees the code and
is told it is wrong, which is much harder.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from .tasks import DEFAULT_IMAGE

REPO_ID = "bigcode/humanevalpack"
JSONL_IN_REPO = "data/python/data/humanevalpack.jsonl"
REQUIRED = ("task_id", "prompt", "declaration", "canonical_solution", "buggy_solution", "test", "entry_point")
TEST_TIMEOUT_S = 30

TEST_SH = """#!/bin/bash
# exit 0 = pass. Runs after the agent finishes; the agent never sees this directory.
set -u
cd "$WORKSPACE"
"${PYTHON:-python3}" "$TESTS_DIR/test_solution.py"
"""

TEST_HEADER = f"""import os
import signal
import sys

# An endless loop in the submitted code is a failed test, not an infrastructure error.
if hasattr(signal, "alarm"):
    signal.alarm({TEST_TIMEOUT_S})

sys.path.insert(0, os.environ.get("WORKSPACE", "/workspace"))
from solution import *  # noqa: E402,F403
"""


def download_jsonl() -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit("huggingface_hub is not installed: run `uv sync --extra infer` (or --extra mlx), "
                         "or pass --input with a local humanevalpack.jsonl") from e
    try:
        return Path(hf_hub_download(REPO_ID, JSONL_IN_REPO, repo_type="dataset"))
    except Exception as e:
        raise SystemExit(f"could not download {JSONL_IN_REPO} from {REPO_ID} ({e}).\n"
                         "Check the network / HF_ENDPOINT, or download the Python split yourself "
                         "and pass it with --input.") from e


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for i, row in enumerate(rows):
        missing = [k for k in REQUIRED if k not in row]
        if missing:
            raise SystemExit(f"row {i} of {path} lacks fields {missing}: not a HumanEvalPack Python file?")
    return rows


def task_name(task_id: str) -> str:
    """'Python/7' -> 'hef_007' (sortable, safe as a directory and container name)."""
    num = re.search(r"(\d+)$", task_id)
    return f"hef_{int(num.group(1)):03d}" if num else re.sub(r"\W+", "_", task_id).lower()


def source_file(row: dict[str, Any], body: str, with_docstring: bool) -> str:
    head = row["prompt"] if with_docstring else row["declaration"]
    imports = row.get("import") or ""
    return (imports + "\n" if imports.strip() else "") + head + body


def render_test_file(row: dict[str, Any]) -> str:
    entry = row["entry_point"]
    body = (row.get("test_setup") or "") + "\n" + row["test"].strip() + "\n"
    if not re.search(rf"^check\({re.escape(entry)}\)", body, re.M):
        body += f"\ncheck({entry})\n"
    return TEST_HEADER + "\n" + body


def instruction(row: dict[str, Any], with_docstring: bool) -> str:
    entry = row["entry_point"]
    spec = ("Its docstring describes the intended behaviour." if with_docstring
            else "Work out the intended behaviour from the code and the function name.")
    return (f"The function `{entry}` in solution.py contains a bug: it returns wrong results or "
            f"does not terminate for some inputs. {spec}\n"
            f"Find the bug and fix it by editing solution.py. Keep the function name and signature.")


def write_task(row: dict[str, Any], out_dir: Path, with_docstring: bool = True) -> Path:
    d = out_dir / task_name(row["task_id"])
    if d.exists():
        shutil.rmtree(d)
    for sub in ("repo", "solution", "tests"):
        (d / sub).mkdir(parents=True)
    (d / "repo" / "solution.py").write_text(source_file(row, row["buggy_solution"], with_docstring))
    (d / "solution" / "solution.py").write_text(source_file(row, row["canonical_solution"], with_docstring))
    (d / "tests" / "test_solution.py").write_text(render_test_file(row))
    (d / "tests" / "test.sh").write_text(TEST_SH)
    meta = {
        "image": DEFAULT_IMAGE,
        "instruction": instruction(row, with_docstring),
        "metadata": {
            "source": f"{REPO_ID}:{row['task_id']}",
            "entry_point": row["entry_point"],
            "bug_type": row.get("bug_type", ""),
            "failure_symptoms": row.get("failure_symptoms", ""),
            "hint": "docstring" if with_docstring else "none",
        },
    }
    (d / "task.yaml").write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
    return d


def convert(rows: list[dict[str, Any]], out_dir: Path, with_docstring: bool = True) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [write_task(row, out_dir, with_docstring) for row in rows]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Convert HumanEvalFix (Python) into mswe-rl-env tasks")
    p.add_argument("--input", help="local humanevalpack.jsonl (Python split); downloaded from the HF hub if omitted")
    p.add_argument("--out", default="tasks/humanevalfix")
    p.add_argument("--hint", choices=["docstring", "none"], default="docstring")
    p.add_argument("--limit", type=int, default=None, help="only convert the first N problems")
    a = p.parse_args(argv)

    rows = read_rows(Path(a.input) if a.input else download_jsonl())
    if a.limit:
        rows = rows[: a.limit]
    dirs = convert(rows, Path(a.out), with_docstring=a.hint == "docstring")
    print(f"wrote {len(dirs)} tasks to {a.out} (hint={a.hint})")
    print(f"next: uv run mswe-check-tasks {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

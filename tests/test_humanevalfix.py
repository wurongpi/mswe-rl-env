import sys
import types
from pathlib import Path

import pytest

from mswe_rl import humanevalfix as hef
from mswe_rl.check_tasks import check_tasks
from mswe_rl.tasks import load_tasks

FIXTURE = Path(__file__).parent / "fixtures" / "humanevalpack_sample.jsonl"


@pytest.fixture(autouse=True)
def short_alarm(monkeypatch):
    # the infinite-loop fixture would otherwise wait the full 30 s
    monkeypatch.setattr(hef, "TEST_HEADER", hef.TEST_HEADER.replace("signal.alarm(30)", "signal.alarm(2)"))


def test_converted_tasks_grade_correctly(tmp_path):
    assert hef.main(["--input", str(FIXTURE), "--out", str(tmp_path / "hef")]) == 0
    tasks = load_tasks(tmp_path / "hef")
    assert [t.name for t in tasks] == ["hef_000", "hef_001", "hef_012"]
    # buggy version fails (incl. the endless loop, via the alarm), reference passes
    assert all(r.ok for r in check_tasks(tasks, backend="local")), check_tasks(tasks, backend="local")
    assert tasks[1].metadata["failure_symptoms"] == "infinite loop"


def test_hint_none_drops_the_docstring(tmp_path):
    rows = hef.read_rows(FIXTURE)
    with_doc = hef.write_task(rows[0], tmp_path / "a", with_docstring=True)
    without = hef.write_task(rows[0], tmp_path / "b", with_docstring=False)
    assert "multiplied by two" in (with_doc / "repo" / "solution.py").read_text()
    assert "multiplied by two" not in (without / "repo" / "solution.py").read_text()
    assert "docstring" not in load_tasks(tmp_path / "b")[0].instruction


def test_check_call_is_added_only_when_missing():
    rows = hef.read_rows(FIXTURE)
    assert hef.render_test_file(rows[0]).count("\ncheck(double)") == 1
    assert hef.render_test_file(rows[1]).rstrip().endswith("check(count_down)")


def test_rejects_files_with_the_wrong_schema(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"task_id": "x"}\n')
    with pytest.raises(SystemExit):
        hef.read_rows(bad)


def test_reads_parquet_via_pyarrow(tmp_path, monkeypatch):
    # pyarrow may not be installed where the tests run; stand in for read_table().to_pylist()
    rows = hef.read_rows(FIXTURE)
    pq = types.ModuleType("pyarrow.parquet")
    pq.read_table = lambda path: types.SimpleNamespace(to_pylist=lambda: rows)
    monkeypatch.setitem(sys.modules, "pyarrow", types.ModuleType("pyarrow"))
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", pq)
    path = tmp_path / "test-00000-of-00001.parquet"
    path.write_bytes(b"")
    assert [r["task_id"] for r in hef.read_rows(path)] == ["Python/0", "Python/1", "Python/12"]

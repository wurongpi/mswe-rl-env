import pytest

from mswe_rl.tasks import load_tasks


def test_toy_tasks_load_with_reference_solutions(tasks):
    assert set(tasks) == {"fix_add", "fix_fizzbuzz", "fix_word_count"}
    for t in tasks.values():
        assert t.instruction and (t.tests_dir / "test.sh").exists()
        assert t.solution_dir is not None


def test_missing_task_set_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tasks(tmp_path / "nope")

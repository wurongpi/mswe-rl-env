import shutil

from mswe_rl.check_tasks import check_task, main
from mswe_rl.tasks import load_task


def test_toy_tasks_grade_correctly(tasks):
    for t in tasks.values():
        r = check_task(t, backend="local")
        assert r.ok, r


def test_detects_tests_that_pass_without_a_fix(tasks, tmp_path):
    # a "bug" that is already fixed: the untouched repo passes -> free reward
    dst = tmp_path / "fix_add"
    shutil.copytree(tasks["fix_add"].repo_dir.parent, dst)
    shutil.copy(dst / "solution" / "calc.py", dst / "repo" / "calc.py")
    r = check_task(load_task(dst), backend="local")
    assert r.buggy_fails is False and not r.ok


def test_detects_broken_reference_solution(tasks, tmp_path):
    dst = tmp_path / "fix_add"
    shutil.copytree(tasks["fix_add"].repo_dir.parent, dst)
    shutil.copy(dst / "repo" / "calc.py", dst / "solution" / "calc.py")
    r = check_task(load_task(dst), backend="local")
    assert r.solution_passes is False and not r.ok


def test_cli_exit_code(tasks, capsys):
    root = next(iter(tasks.values())).repo_dir.parent.parent
    assert main([str(root)]) == 0
    assert "3/3 tasks grade correctly" in capsys.readouterr().out

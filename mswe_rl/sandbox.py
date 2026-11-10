"""Sandboxes that mini-swe-agent can drive.

mini-swe-agent only needs an environment object with
    execute(command, cwd="", timeout=None) -> {"output": str, "returncode": int}
    get_template_vars() -> dict
so we implement that protocol ourselves instead of depending on its Docker
class. That keeps container lifecycle, test injection and failure
classification under our control.

Backends:
  DockerSandbox  - one container per episode. Works with OrbStack, Colima or
                   Docker Desktop: anything that provides the `docker` CLI.
  LocalSandbox   - a temp directory on the host. No isolation: only for
                   debugging the pipeline and for the test suite.

Test files are copied in only AFTER the agent finishes, so the policy cannot
read or edit them (a common reward-hacking route).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Protocol


class EnvError(RuntimeError):
    """Infrastructure failure: not the policy's fault, exclude from training."""


class Sandbox(Protocol):
    """What an episode needs from a sandbox. Both backends implement it."""

    workdir: str

    def start(self) -> None: ...

    def execute(self, command: Any, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Run a shell command. Returns {"output": str, "returncode": int}."""

    def get_template_vars(self) -> dict[str, Any]: ...

    def write_files(self, src_dir: Path) -> None:
        """Copy the contents of src_dir over the workspace (used to apply reference solutions)."""

    def run_tests(self, tests_dir: Path, timeout: int = 120) -> tuple[bool, str]:
        """Inject the hidden tests and run tests/test.sh. Returns (passed, log tail)."""

    def stop(self) -> None: ...


TEMPLATE_VARS = {"system": "Linux", "release": "", "version": "", "machine": "aarch64"}


def _coerce_command(command) -> str:
    # Be tolerant of harness versions that pass an action dict.
    if isinstance(command, dict):
        return command.get("command") or command.get("action") or ""
    return command


class DockerSandbox:
    def __init__(
        self,
        image: str,
        repo_dir: Path,
        workdir: str = "/workspace",
        default_timeout: int = 60,
        docker: str = "docker",
    ):
        self.image = image
        self.repo_dir = Path(repo_dir)
        self.workdir = workdir
        self.default_timeout = default_timeout
        self.docker = docker
        self.name = f"mswe-{uuid.uuid4().hex[:10]}"
        self.started = False

    def _run(self, args: list[str], timeout: int | None = 120, check: bool = True):
        try:
            r = subprocess.run([self.docker, *args], capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError as e:
            raise EnvError("docker CLI not found (install OrbStack or Colima)") from e
        except subprocess.TimeoutExpired as e:
            raise EnvError(f"docker {args[0]} timed out") from e
        if check and r.returncode != 0:
            raise EnvError(f"docker {' '.join(args[:2])} failed: {r.stderr.strip()[:500]}")
        return r

    def start(self):
        self._run(
            [
                "run",
                "-d",
                "--rm",
                "--name",
                self.name,
                "-w",
                self.workdir,
                "--network",
                "none",
                "--memory",
                "2g",
                self.image,
                "sleep",
                "infinity",
            ],
            timeout=300,
        )
        self.started = True
        self._run(["exec", self.name, "mkdir", "-p", self.workdir])
        self._run(["cp", f"{self.repo_dir}/.", f"{self.name}:{self.workdir}"])

    # --- mini-swe-agent environment protocol ---
    def execute(self, command, cwd: str = "", *, timeout: int | None = None, **_):
        command = _coerce_command(command)
        timeout = timeout or self.default_timeout
        try:
            r = subprocess.run(
                [self.docker, "exec", "-w", cwd or self.workdir, self.name, "bash", "-lc", command],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            # A slow command is the policy's problem: report it as an observation.
            return {"output": f"Command timed out after {timeout}s", "returncode": 124}
        if r.returncode in (125, 126) or "No such container" in r.stderr:
            raise EnvError(f"container lost: {r.stderr.strip()[:300]}")
        return {"output": r.stdout + r.stderr, "returncode": r.returncode}

    def get_template_vars(self) -> dict:
        return {**TEMPLATE_VARS, "cwd": self.workdir}

    def write_files(self, src_dir: Path) -> None:
        self._run(["cp", f"{Path(src_dir)}/.", f"{self.name}:{self.workdir}"])

    # --- reward ---
    def run_tests(self, tests_dir: Path, timeout: int = 120) -> tuple[bool, str]:
        self._run(["cp", f"{Path(tests_dir)}/.", f"{self.name}:/tests"])
        try:
            r = subprocess.run(
                [
                    self.docker,
                    "exec",
                    "-e",
                    f"WORKSPACE={self.workdir}",
                    "-e",
                    "TESTS_DIR=/tests",
                    self.name,
                    "bash",
                    "/tests/test.sh",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise EnvError("test run timed out") from e
        if r.returncode in (125, 126) or "No such container" in r.stderr:
            raise EnvError(f"container lost during tests: {r.stderr.strip()[:300]}")
        return r.returncode == 0, (r.stdout + r.stderr)[-2000:]

    def stop(self):
        if self.started:
            self._run(["rm", "-f", self.name], check=False)
            self.started = False


class LocalSandbox:
    """Runs commands in a temp dir on the host. NOT isolated - debugging only."""

    def __init__(self, image: str, repo_dir: Path, default_timeout: int = 60, **_):
        self.repo_dir = Path(repo_dir)
        self.default_timeout = default_timeout
        self.root: Path | None = None
        self.workdir = ""

    def start(self):
        self.root = Path(tempfile.mkdtemp(prefix="mswe-"))
        ws = self.root / "workspace"
        shutil.copytree(self.repo_dir, ws)
        self.workdir = str(ws)

    def execute(self, command, cwd: str = "", *, timeout: int | None = None, **_):
        command = _coerce_command(command)
        timeout = timeout or self.default_timeout
        try:
            r = subprocess.run(
                ["bash", "-c", command],
                cwd=cwd or self.workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"output": f"Command timed out after {timeout}s", "returncode": 124}
        return {"output": r.stdout + r.stderr, "returncode": r.returncode}

    def get_template_vars(self) -> dict:
        import platform

        return {**platform.uname()._asdict(), "cwd": self.workdir}

    def write_files(self, src_dir: Path) -> None:
        shutil.copytree(src_dir, self.workdir, dirs_exist_ok=True)

    def run_tests(self, tests_dir: Path, timeout: int = 120) -> tuple[bool, str]:
        assert self.root is not None
        dst = self.root / "tests"
        shutil.copytree(tests_dir, dst, dirs_exist_ok=True)
        env = {**os.environ, "WORKSPACE": self.workdir, "TESTS_DIR": str(dst)}
        try:
            r = subprocess.run(
                ["bash", str(dst / "test.sh")], capture_output=True, text=True, timeout=timeout, env=env
            )
        except subprocess.TimeoutExpired as e:
            raise EnvError("test run timed out") from e
        return r.returncode == 0, (r.stdout + r.stderr)[-2000:]

    def stop(self):
        if self.root and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)


def make_sandbox(backend: str, image: str, repo_dir: Path, **kw) -> Sandbox:
    if backend == "docker":
        return DockerSandbox(image, repo_dir, **kw)
    if backend == "local":
        return LocalSandbox(image, repo_dir, **kw)
    raise ValueError(f"unknown backend {backend}")

import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(app) -> str:
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}"


@pytest.fixture(scope="session")
def fake_model_url():
    from mswe_rl.testing.fake_server import build_app

    scripts = yaml.safe_load((ROOT / "demo" / "scripts.yaml").read_text())
    return _serve(build_app(scripts))


@pytest.fixture(scope="session")
def proxy_url(fake_model_url, tmp_path_factory):
    from mswe_rl.proxy import build_app

    return _serve(build_app(fake_model_url, record_dir=str(tmp_path_factory.mktemp("calls"))))


@pytest.fixture
def tasks():
    from mswe_rl.tasks import load_tasks

    return {t.name: t for t in load_tasks(ROOT / "tasks" / "toy")}

"""Run mini-swe-agent (v2.x, pinned as a git submodule) against our sandbox.

Two adapters plug our pieces into mini-swe-agent's protocols:

  SandboxEnv      - wraps DockerSandbox / LocalSandbox as a mini-swe-agent
                    Environment (execute(action) / get_template_vars / serialize)
                    and raises Submitted when the agent runs the submit command.

  ProxyTextModel  - a mini-swe-agent Model that talks to the recording proxy
                    with plain HTTP. It reuses mini-swe-agent's own text-action
                    parser and observation formatter, so the agent behaves
                    exactly like `mini` with the `mini_textbased.yaml` config.
                    Text-based actions (```mswea_bash_command blocks) work with
                    any small local model; no tool-calling support needed.

Pass model_backend="litellm" to use mini-swe-agent's LitellmTextbasedModel
instead (same prompts, goes through litellm).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests
import yaml

from . import SUBMIT_MARKER

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")


class SandboxEnv:
    def __init__(self, sandbox, extra_env: dict[str, str] | None = None):
        self.sandbox = sandbox
        self.config = {"backend": type(sandbox).__name__}
        self.extra_env = extra_env or {}

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        from minisweagent.exceptions import Submitted

        command = action.get("command", "") if isinstance(action, dict) else str(action)
        if self.extra_env:
            exports = " ".join(f"{k}={v!r}" for k, v in self.extra_env.items())
            command = f"export {exports}; {command}"
        out = self.sandbox.execute(command, cwd, timeout=timeout)  # EnvError propagates
        out.setdefault("exception_info", "")
        lines = out.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == SUBMIT_MARKER and out["returncode"] == 0:
            sub = "".join(lines[1:])
            raise Submitted(
                {"role": "exit", "content": sub, "extra": {"exit_status": "Submitted", "submission": sub}}
            )
        return out

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return {**self.sandbox.get_template_vars(), **kwargs}

    def serialize(self) -> dict:
        return {"info": {"config": {"environment": self.config}}}


class ProxyTextModel:
    def __init__(
        self,
        *,
        api_base: str,
        model_name: str,
        model_kwargs: dict | None = None,
        action_regex: str = r"```mswea_bash_command\s*\n(.*?)\n```",
        observation_template: str = "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>",
        format_error_template: str = "{{ error }}",
        timeout: float = 600,
        **_,
    ):
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.model_kwargs = {k: v for k, v in (model_kwargs or {}).items() if k != "drop_params"}
        self.action_regex = action_regex
        self.observation_template = observation_template
        self.format_error_template = format_error_template
        self.timeout = timeout
        self.config = {"model_name": model_name, "api_base": self.api_base}

    def query(self, messages: list[dict], **kwargs) -> dict:
        from minisweagent.exceptions import FormatError
        from minisweagent.models.utils.actions_text import parse_regex_actions

        payload = {
            "model": self.model_name,
            "messages": [{"role": m["role"], "content": m.get("content") or ""} for m in messages],
            **self.model_kwargs,
            **kwargs,
        }
        r = requests.post(f"{self.api_base}/chat/completions", json=payload, timeout=self.timeout)
        r.raise_for_status()
        resp = r.json()
        choice = resp["choices"][0]
        content = choice["message"].get("content") or ""
        extra = {"cost": 0.0, "timestamp": time.time(), "response": resp}
        try:
            actions = parse_regex_actions(
                content,
                action_regex=self.action_regex,
                format_error_template=self.format_error_template,
                template_kwargs={"finish_reason": choice.get("finish_reason")},
            )
        except FormatError as e:
            # mini-swe-agent would add only the error message to the history and drop the
            # model's reply, which breaks the token prefix of the next call. Keep the reply.
            e.messages = ({"role": "assistant", "content": content, "extra": extra}, *e.messages)
            raise
        return {"role": "assistant", "content": content, "extra": {"actions": actions, **extra}}

    def format_message(self, **kwargs) -> dict:
        return kwargs

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ):
        from minisweagent.models.utils.actions_text import format_observation_messages

        return format_observation_messages(
            outputs, observation_template=self.observation_template, template_vars=template_vars
        )

    def get_template_vars(self, **kwargs) -> dict:
        return {"model_name": self.model_name, **kwargs}

    def serialize(self) -> dict:
        return {"info": {"config": {"model": self.config}}}


def load_mini_config(config_path: str | None = None) -> dict:
    from minisweagent.config import builtin_config_dir

    path = Path(config_path) if config_path else Path(builtin_config_dir) / "mini_textbased.yaml"
    return yaml.safe_load(path.read_text()) or {}


def run_mini_swe_agent(
    task_text: str,
    sandbox,
    api_base: str,
    model_name: str,
    step_limit: int = 15,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    config_path: str | None = None,
    model_backend: str = "proxy",
    trajectory_path: str | None = None,
) -> str:
    from minisweagent.agents.default import DefaultAgent

    cfg = load_mini_config(config_path)
    agent_cfg = {k: v for k, v in cfg.get("agent", {}).items() if k != "mode"}
    agent_cfg.update(step_limit=step_limit, cost_limit=0)
    if trajectory_path:
        agent_cfg["output_path"] = Path(trajectory_path)
    model_cfg = dict(cfg.get("model", {}))
    model_cfg["model_kwargs"] = {
        **model_cfg.get("model_kwargs", {}),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if model_backend == "litellm":
        from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

        model_cfg["model_kwargs"].update(api_base=api_base, api_key="local")
        model = LitellmTextbasedModel(
            model_name=f"openai/{model_name}", cost_tracking="ignore_errors", **model_cfg
        )
    else:
        model = ProxyTextModel(api_base=api_base, model_name=model_name, **model_cfg)

    env = SandboxEnv(sandbox, cfg.get("environment", {}).get("env"))
    agent = DefaultAgent(model, env, **agent_cfg)
    result = agent.run(task_text)
    return str(result.get("exit_status", ""))

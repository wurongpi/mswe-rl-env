"""Run mini-swe-agent against a sandbox, with the model pointed at the recording proxy.

This is the only file that imports mini-swe-agent. Its public API has been
stable across 1.x (DefaultAgent, LitellmModel, builtin_config_dir), but if a
future release renames something, this is the one place to fix.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def run_mini_swe_agent(task_text: str, sandbox, api_base: str, model_name: str,
                       step_limit: int = 15, temperature: float = 1.0, max_tokens: int = 1024,
                       config_path: str | None = None) -> str:
    # Local models have no litellm price entry; don't let cost tracking crash the run.
    os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.config import builtin_config_dir
    from minisweagent.models.litellm_model import LitellmModel

    cfg_file = Path(config_path) if config_path else Path(builtin_config_dir) / "default.yaml"
    cfg = yaml.safe_load(cfg_file.read_text()) or {}
    agent_cfg = dict(cfg.get("agent", {}))
    agent_cfg["step_limit"] = step_limit
    agent_cfg["cost_limit"] = 0  # 0 = no limit

    model = LitellmModel(
        model_name=f"openai/{model_name}",
        model_kwargs={
            "api_base": api_base,
            "api_key": "local",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "drop_params": True,
        },
    )
    agent = DefaultAgent(model, sandbox, **agent_cfg)
    exit_status, _ = agent.run(task_text)
    return str(exit_status)

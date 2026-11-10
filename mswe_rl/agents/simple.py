"""A ~60-line bash-only agent with the same action format as mini-swe-agent's text-based mode.

Used by the test suite and the offline demo (no mini-swe-agent install needed),
and as a readable reference for what the harness does each step:

    model reply -> first ```mswea_bash_command block -> run in sandbox -> observation as next user turn

`max_history` keeps only the system prompt, the task and the last N messages.
That is exactly the kind of history truncation that breaks the token prefix
between calls, so the demo uses it to show segment splitting.
"""

from __future__ import annotations

import re

import requests

from . import SUBMIT_MARKER

SYSTEM = (
    "You are a software engineer working in a Linux shell. Reply with a short THOUGHT and "
    "exactly one ```mswea_bash_command code block containing one command. When the task is done, run "
    f"`echo {SUBMIT_MARKER}`."
)

BASH_RE = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.S)


def run_simple_agent(
    task_text: str,
    sandbox,
    api_base: str,
    model_name: str = "local",
    step_limit: int = 15,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    max_history: int | None = None,
    timeout: float = 600,
) -> str:
    head = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Working directory: {sandbox.get_template_vars()['cwd']}\n\nTask:\n{task_text}",
        },
    ]
    tail: list[dict] = []
    for _ in range(step_limit):
        window = tail if max_history is None else tail[-max_history:]
        r = requests.post(
            f"{api_base}/chat/completions",
            json={
                "model": model_name,
                "messages": head + window,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"]
        tail.append({"role": "assistant", "content": reply})

        m = BASH_RE.search(reply)
        if not m:
            tail.append(
                {
                    "role": "user",
                    "content": "Format error: reply with exactly one ```mswea_bash_command block.",
                }
            )
            continue
        cmd = m.group(1).strip()
        out = sandbox.execute(cmd)
        if out["output"].lstrip().startswith(SUBMIT_MARKER):
            return "Submitted"
        tail.append(
            {
                "role": "user",
                "content": f"<returncode>{out['returncode']}</returncode>\n<output>\n{out['output'][-4000:]}</output>",
            }
        )
    return "LimitsExceeded"

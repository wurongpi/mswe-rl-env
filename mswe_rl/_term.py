"""Tiny ANSI helper shared by the command-line tools."""

from __future__ import annotations

import sys

_CODES = {"green": "32", "red": "31", "yellow": "33", "gray": "90", "dim": "2", "bold": "1"}


class Style:
    """``Style(force)`` colors text when stdout is a terminal or ``force`` is set."""

    def __init__(self, force: bool = False):
        self.enabled = force or sys.stdout.isatty()

    def __call__(self, name: str, text: str) -> str:
        return f"\033[{_CODES[name]}m{text}\033[0m" if self.enabled else text

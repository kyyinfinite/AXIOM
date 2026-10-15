from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AgentState": "axiom.tui.state",
    "InteractiveSession": "axiom.tui.session",
    "TerminalRenderer": "axiom.tui.renderer",
    "chat_async": "axiom.tui.app",
    "run_interactive": "axiom.tui.app",
    "run_tui": "axiom.tui.app",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(name)
    return getattr(import_module(module), name)

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AgentState": "eurus.tui.state",
    "InteractiveSession": "eurus.tui.session",
    "TerminalRenderer": "eurus.tui.renderer",
    "chat_async": "eurus.tui.app",
    "run_interactive": "eurus.tui.app",
    "run_tui": "eurus.tui.app",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(name)
    return getattr(import_module(module), name)

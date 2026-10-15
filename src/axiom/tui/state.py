from __future__ import annotations

from enum import Enum


class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    RUNNING = "running"
    APPROVAL = "approval"
    ERROR = "error"

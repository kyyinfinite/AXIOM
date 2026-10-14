from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from axiom.models import ToolCall, ToolResult, Usage

AgentEventKind = Literal["step", "text", "tool_call", "tool_result", "done"]


@dataclass(frozen=True, slots=True)
class AgentEvent:
    kind: AgentEventKind
    step: int = 0
    text: str = ""
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    usage: Usage | None = None
    stop_reason: str = ""

    @classmethod
    def step_started(cls, step: int) -> AgentEvent:
        return cls("step", step=step)

    @classmethod
    def text_delta(cls, text: str) -> AgentEvent:
        return cls("text", text=text)

    @classmethod
    def tool_started(cls, call: ToolCall) -> AgentEvent:
        return cls("tool_call", tool_call=call)

    @classmethod
    def tool_finished(cls, result: ToolResult) -> AgentEvent:
        return cls("tool_result", tool_result=result)

    @classmethod
    def finished(cls, stop_reason: str, usage: Usage) -> AgentEvent:
        return cls("done", stop_reason=stop_reason, usage=usage)

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axiom.errors import ToolError
from axiom.models import ToolCall, ToolResult
from axiom.providers.base import INVALID_ARGUMENTS_KEY, ToolSpec
from axiom.tools.schema import validate_arguments

NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
HEAD_SHARE = 0.7

ToolHandler = Callable[[dict[str, Any], "ToolContext"], Awaitable[str]]


@dataclass(frozen=True, slots=True)
class ToolContext:
    workspace: Path
    max_output: int = 20000
    timeout: float = 120.0


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    mutates: bool = False
    requires_approval: bool = False

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)


def truncate_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    head = int(limit * HEAD_SHARE)
    tail = limit - head
    removed = len(text) - limit
    ending = text[-tail:] if tail else ""
    return f"{text[:head]}\n... [{removed} characters truncated] ...\n{ending}"


def _error(call: ToolCall, message: str, limit: int) -> ToolResult:
    return ToolResult(call.id, call.name, truncate_text(f"error: {message}", limit), True)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for item in tools:
            self.register(item)

    def register(self, tool: Tool) -> None:
        if not NAME_PATTERN.match(tool.name):
            raise ValueError(f"invalid tool name: {tool.name!r}")
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        *,
        mutates: bool = False,
        requires_approval: bool = False,
    ) -> Callable[[ToolHandler], ToolHandler]:
        def decorator(handler: ToolHandler) -> ToolHandler:
            self.register(Tool(name, description, parameters, handler, mutates, requires_approval))
            return handler

        return decorator

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> list[Tool]:
        return [self._tools[name] for name in self.names()]

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(tool.spec() for tool in self.all())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        limit = context.max_output
        tool = self._tools.get(call.name)
        if tool is None:
            known = ", ".join(self.names()) or "none"
            return _error(call, f"unknown tool '{call.name}'. available tools: {known}", limit)
        if INVALID_ARGUMENTS_KEY in call.arguments:
            return _error(call, "arguments are not valid JSON, send a JSON object", limit)
        problems = validate_arguments(tool.parameters, call.arguments)
        if problems:
            return _error(call, "invalid arguments: " + "; ".join(problems), limit)
        try:
            output = await asyncio.wait_for(
                tool.handler(dict(call.arguments), context), timeout=context.timeout
            )
        except ToolError as exc:
            return _error(call, str(exc), limit)
        except TimeoutError:
            return _error(call, f"timed out after {context.timeout:g}s", limit)
        except Exception as exc:
            return _error(call, f"tool crashed: {type(exc).__name__}: {exc}", limit)
        return ToolResult(call.id, call.name, truncate_text(output, limit))

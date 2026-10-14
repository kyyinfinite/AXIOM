from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable

from axiom.agent.events import AgentEvent
from axiom.models import Message, ToolCall, ToolResult, Usage
from axiom.providers.router import Router
from axiom.tools.base import Tool, ToolContext, ToolRegistry

REPEAT_LIMIT = 3
STOP_REASONS = {
    "stop": "completed",
    "tool_calls": "completed",
    "length": "truncated",
    "content_filter": "filtered",
}

Approver = Callable[[ToolCall, Tool], Awaitable[bool]]


class Agent:
    def __init__(
        self,
        router: Router,
        registry: ToolRegistry,
        context: ToolContext,
        *,
        system_prompt: str,
        max_steps: int = 40,
        approver: Approver | None = None,
        stream: bool = True,
        max_parallel_tools: int = 4,
    ) -> None:
        self.router = router
        self.registry = registry
        self.context = context
        self.max_steps = max_steps
        self.approver = approver
        self.stream = stream
        if max_parallel_tools < 1:
            raise ValueError("max_parallel_tools must be at least 1")
        self.max_parallel_tools = max_parallel_tools
        self._system_prompt = system_prompt
        self.history: list[Message] = [Message.system(system_prompt)]
        self.usage = Usage()

    def reset(self) -> None:
        self.history = [Message.system(self._system_prompt)]
        self.usage = Usage()

    async def run(self, text: str) -> AsyncIterator[AgentEvent]:
        self.history.append(Message.user(text))
        recent: list[str] = []
        for step in range(1, self.max_steps + 1):
            yield AgentEvent.step_started(step)
            parts: list[str] = []
            calls: list[ToolCall] = []
            finish = "stop"
            specs = self.registry.specs()
            if self.stream:
                async for event in self.router.stream(self.history, tools=specs):
                    if event.kind == "text":
                        parts.append(event.text)
                        yield AgentEvent.text_delta(event.text)
                    elif event.kind == "tool_call" and event.tool_call is not None:
                        calls.append(event.tool_call)
                    elif event.kind == "usage" and event.usage is not None:
                        self.usage += event.usage
                    elif event.kind == "done":
                        finish = event.finish_reason or finish
            else:
                response = await self.router.chat(self.history, tools=specs)
                if response.message.content:
                    parts.append(response.message.content)
                    yield AgentEvent.text_delta(response.message.content)
                calls = list(response.message.tool_calls)
                finish = response.finish_reason
                self.usage += response.usage
            content = "".join(parts)
            if content or calls:
                self.history.append(Message.assistant(content, tuple(calls)))
            if not calls:
                yield AgentEvent.finished(STOP_REASONS.get(finish, "completed"), self.usage)
                return
            async for tool_event in self._run_tools(calls, recent):
                yield tool_event
        yield AgentEvent.finished("max_steps", self.usage)

    async def _run_tools(
        self, calls: list[ToolCall], recent: list[str]
    ) -> AsyncIterator[AgentEvent]:
        prepared: list[tuple[ToolCall, Tool | None, ToolResult | None]] = []
        for call in calls:
            tool, rejected = await self._prepare(call, recent)
            prepared.append((call, tool, rejected))
            yield AgentEvent.tool_started(call)

        semaphore = asyncio.Semaphore(self.max_parallel_tools)

        async def run_one(
            call: ToolCall, tool: Tool | None, rejected: ToolResult | None
        ) -> ToolResult:
            if rejected is not None:
                return rejected
            if tool is None:
                return await self.registry.execute(call, self.context)
            async with semaphore:
                return await self.registry.execute(call, self.context)

        outcomes = await asyncio.gather(
            *(run_one(call, tool, rejected) for call, tool, rejected in prepared),
            return_exceptions=True,
        )
        interrupted: BaseException | None = None
        results: list[ToolResult] = []
        for (call, _, _), outcome in zip(prepared, outcomes, strict=True):
            if isinstance(outcome, ToolResult):
                results.append(outcome)
            else:
                if interrupted is None and isinstance(outcome, BaseException):
                    interrupted = outcome
                results.append(
                    ToolResult(call.id, call.name, "error: cancelled before finishing", True)
                )

        for result in results:
            self.history.append(result.to_message())
            yield AgentEvent.tool_finished(result)

        if interrupted is not None:
            raise interrupted

    async def _prepare(
        self, call: ToolCall, recent: list[str]
    ) -> tuple[Tool | None, ToolResult | None]:
        arguments = json.dumps(call.arguments, sort_keys=True, default=str)
        recent.append(f"{call.name}:{arguments}")
        if len(recent) >= REPEAT_LIMIT and len(set(recent[-REPEAT_LIMIT:])) == 1:
            return None, ToolResult(
                call.id,
                call.name,
                "error: this exact call was repeated too many times, change your approach",
                True,
            )
        tool = self.registry.get(call.name)
        if tool is None:
            return None, None
        if tool.mutates or tool.requires_approval:
            if self.approver is None:
                return None, ToolResult(
                    call.id,
                    call.name,
                    "error: this action requires owner approval, but "
                    "no approval provider is configured",
                    True,
                )
            if not await self.approver(call, tool):
                return None, ToolResult(
                    call.id, call.name, "error: the user denied this action", True
                )
        return tool, None

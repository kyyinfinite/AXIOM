import asyncio
import contextlib
from pathlib import Path

from eurus.agent.loop import Agent, IterationBudget
from eurus.providers.base import StreamEvent
from eurus.providers.testing import openai_stream, scripted_router
from eurus.tools.base import Tool, ToolContext, ToolRegistry

EMPTY = {"type": "object", "properties": {}}


def test_iteration_budget_enforces_limit_and_reports_remaining():
    budget = IterationBudget()
    assert budget.limit == 500
    assert budget.remaining == 500
    assert budget.consume() is True
    assert budget.used == 1
    limited = IterationBudget(limit=2)
    assert [limited.consume() for _ in range(3)] == [True, True, False]
    assert limited.remaining == 0


def test_tool_defaults_preserve_parallel_read_only_behavior():
    async def handler(arguments, context):
        return "ok"

    tool = Tool("read", "read", EMPTY, handler)
    assert tool.parallel_safe is True
    assert tool.interactive is False
    assert tool.path_keys == ()


def test_same_mutating_path_is_serialized():
    async def run():
        active = 0
        peak = 0

        async def writer(arguments, context):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.03)
            active -= 1
            return arguments["path"]

        registry = ToolRegistry(
            [
                Tool(
                    "write_path",
                    "write path",
                    {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                    writer,
                    mutates=True,
                    path_keys=("path",),
                )
            ]
        )

        async def approve(call, tool):
            return True

        router, _ = scripted_router(
            streams=[
                openai_stream(
                    "",
                    [
                        ("a", "write_path", {"path": "same.txt"}),
                        ("b", "write_path", {"path": "same.txt"}),
                    ],
                ),
                openai_stream("done"),
            ]
        )
        agent = Agent(
            router,
            registry,
            ToolContext(workspace=Path(".")),
            system_prompt="sys",
            approver=approve,
        )
        events = [event async for event in agent.run("go")]
        assert peak == 1
        assert [event.tool_result.output for event in events if event.tool_result] == [
            "same.txt",
            "same.txt",
        ]

    asyncio.run(run())


def test_interactive_tools_run_sequentially():
    async def run():
        active = 0
        peak = 0

        async def handler(arguments, context):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return arguments["value"]

        registry = ToolRegistry(
            [
                Tool(
                    "prompt",
                    "prompt",
                    {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                    handler,
                    interactive=True,
                )
            ]
        )
        router, _ = scripted_router(
            streams=[
                openai_stream(
                    "",
                    [("a", "prompt", {"value": "a"}), ("b", "prompt", {"value": "b"})],
                ),
                openai_stream("done"),
            ]
        )
        agent = Agent(router, registry, ToolContext(workspace=Path(".")), system_prompt="sys")
        [event async for event in agent.run("go")]
        assert peak == 1

    asyncio.run(run())


def test_interrupt_does_not_commit_partial_assistant_message():
    async def run():
        class SlowRouter:
            async def chat(self, messages, tools=()):
                await asyncio.sleep(0.1)
                raise AssertionError("streaming is expected")

            async def stream(self, messages, tools=()):
                yield StreamEvent.text_delta("partial")
                await asyncio.sleep(0.1)

        agent = Agent(
            SlowRouter(),
            ToolRegistry(),
            ToolContext(workspace=Path(".")),
            system_prompt="sys",
        )
        task = asyncio.create_task(_collect(agent))
        await asyncio.sleep(0.01)
        agent.interrupt()
        events = await task
        assert events[-1].stop_reason == "interrupted"
        assert any(event.kind == "text" and event.text == "partial" for event in events)
        assert [message.role for message in agent.history] == ["system", "user"]

    async def _collect(agent):
        return [event async for event in agent.run("go")]

    asyncio.run(run())


def test_api_timeout_does_not_commit_partial_assistant_message():
    async def run():
        class SlowRouter:
            async def chat(self, messages, tools=()):
                await asyncio.sleep(0.1)
                raise AssertionError("must time out first")

            async def stream(self, messages, tools=()):
                yield StreamEvent.text_delta("partial")
                await asyncio.sleep(0.1)

        agent = Agent(
            SlowRouter(),
            ToolRegistry(),
            ToolContext(workspace=Path(".")),
            system_prompt="sys",
            request_timeout=0.01,
        )
        events = [event async for event in agent.run("go")]
        assert events[-1].stop_reason == "timeout"
        assert any(event.kind == "text" and event.text == "partial" for event in events)
        assert [message.role for message in agent.history] == ["system", "user"]

    asyncio.run(run())


def test_external_cancellation_closes_active_stream_task():
    async def run():
        started = asyncio.Event()
        cleaned = asyncio.Event()

        class SlowRouter:
            async def chat(self, messages, tools=()):
                raise AssertionError("streaming is expected")

            async def stream(self, messages, tools=()):
                started.set()
                try:
                    yield StreamEvent.text_delta("partial")
                    await asyncio.Event().wait()
                finally:
                    cleaned.set()

        agent = Agent(
            SlowRouter(),
            ToolRegistry(),
            ToolContext(workspace=Path(".")),
            system_prompt="sys",
        )
        task = asyncio.create_task(_collect(agent))
        await started.wait()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert cleaned.is_set()

    async def _collect(agent):
        return [event async for event in agent.run("go")]

    asyncio.run(run())

import asyncio
from datetime import date
from pathlib import Path

import pytest

from eurus.agent import Agent
from eurus.agent.prompt import build_system_prompt
from eurus.models import ToolResult
from eurus.providers.testing import openai_message, openai_stream, scripted_router
from eurus.tools import Tool, ToolContext, ToolRegistry, default_registry

EMPTY = {"type": "object", "properties": {}}


class Stop(BaseException):
    pass


def make_agent(router, registry=None, approver=None, stream=True, max_steps=10):
    context = ToolContext(workspace=Path("."), max_output=1000, timeout=5.0)
    return Agent(
        router,
        registry or default_registry(),
        context,
        system_prompt="sys",
        max_steps=max_steps,
        approver=approver,
        stream=stream,
    )


def run_agent(agent, text="go"):
    async def go():
        return [event async for event in agent.run(text)]

    return asyncio.run(go())


def kinds(events):
    return [event.kind for event in events]


def calc(call_id, expression):
    return (call_id, "calculate", {"expression": expression})


def test_plain_answer():
    router, transport = scripted_router(streams=[openai_stream("hello")])
    agent = make_agent(router)
    events = run_agent(agent)
    assert kinds(events) == ["step", "text", "done"]
    assert events[-1].stop_reason == "completed"
    assert events[-1].usage.total_tokens == 15
    assert [m.role for m in agent.history] == ["system", "user", "assistant"]
    tools = transport.calls[0].payload["tools"]
    assert [t["function"]["name"] for t in tools] == ["calculate", "get_time"]


def test_tool_round_trip_streaming():
    router, transport = scripted_router(
        streams=[openai_stream("", [calc("c1", "2+3*4")]), openai_stream("It is 14")]
    )
    agent = make_agent(router)
    events = run_agent(agent, "what is 2+3*4")
    assert kinds(events) == ["step", "tool_call", "tool_result", "step", "text", "done"]
    assert events[2].tool_result.output == "14"
    assert events[-1].usage.total_tokens == 30
    second = transport.calls[1].payload["messages"]
    assert second[-1] == {"role": "tool", "tool_call_id": "c1", "content": "14"}
    assert second[-2]["tool_calls"][0]["id"] == "c1"
    assert [m.role for m in agent.history] == ["system", "user", "assistant", "tool", "assistant"]


def test_tool_round_trip_without_streaming():
    router, _ = scripted_router(
        json_responses=[openai_message("", [calc("c1", "6*7")]), openai_message("42")]
    )
    events = run_agent(make_agent(router, stream=False))
    assert kinds(events) == ["step", "tool_call", "tool_result", "step", "text", "done"]
    assert events[2].tool_result.output == "42"
    assert events[4].text == "42"


def test_unknown_tool_is_reported_to_the_model():
    router, transport = scripted_router(
        streams=[openai_stream("", [("c1", "nope", {})]), openai_stream("sorry")]
    )
    events = run_agent(make_agent(router))
    result = events[2].tool_result
    assert result.is_error
    assert "unknown tool" in result.output
    assert events[-1].stop_reason == "completed"
    assert "unknown tool" in transport.calls[1].payload["messages"][-1]["content"]


def test_invalid_arguments_are_reported():
    router, _ = scripted_router(
        streams=[openai_stream("", [("c1", "calculate", {})]), openai_stream("ok")]
    )
    events = run_agent(make_agent(router))
    assert "expression is required" in events[2].tool_result.output


def test_unparseable_arguments_are_reported():
    body = openai_message("", [calc("c1", "1+1")])
    body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "oops"
    router, _ = scripted_router(json_responses=[body, openai_message("done")])
    events = run_agent(make_agent(router, stream=False))
    assert "not valid JSON" in events[2].tool_result.output


def test_multiple_tool_calls_in_one_step_keep_order():
    router, transport = scripted_router(
        streams=[
            openai_stream("", [calc("a", "1+1"), calc("b", "2+2")]),
            openai_stream("done"),
        ]
    )
    events = run_agent(make_agent(router))
    outputs = [e.tool_result.output for e in events if e.kind == "tool_result"]
    assert outputs == ["2", "4"]
    tool_messages = [m for m in transport.calls[1].payload["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["a", "b"]


def test_max_steps_stops_the_loop():
    router, _ = scripted_router(
        streams=[openai_stream("", [calc(f"c{i}", f"{i}+1")]) for i in range(3)]
    )
    events = run_agent(make_agent(router, max_steps=3))
    assert events[-1].kind == "done"
    assert events[-1].stop_reason == "max_steps"
    assert kinds(events).count("tool_result") == 3


def test_identical_calls_are_blocked_after_two_repeats():
    streams = [openai_stream("", [calc(f"c{i}", "1+1")]) for i in range(4)]
    streams.append(openai_stream("giving up"))
    router, _ = scripted_router(streams=streams)
    events = run_agent(make_agent(router))
    results = [e.tool_result for e in events if e.kind == "tool_result"]
    assert [r.is_error for r in results] == [False, False, True, True]
    assert "repeated too many times" in results[2].output
    assert events[-1].stop_reason == "completed"


def mutating_registry(calls):
    async def write(arguments, ctx):
        calls.append(arguments)
        return "written"

    registry = default_registry()
    registry.register(Tool("write", "write something", EMPTY, write, mutates=True))
    return registry


def test_approver_can_deny_mutating_tools():
    calls = []
    asked = []

    async def approver(call, tool):
        asked.append(call.name)
        return False

    router, _ = scripted_router(
        streams=[openai_stream("", [("c1", "write", {})]), openai_stream("ok")]
    )
    events = run_agent(make_agent(router, mutating_registry(calls), approver))
    assert asked == ["write"]
    assert calls == []
    assert events[2].tool_result.is_error
    assert "denied" in events[2].tool_result.output


def test_approver_allows_and_skips_read_only_tools():
    calls = []
    asked = []

    async def approver(call, tool):
        asked.append(call.name)
        return True

    router, _ = scripted_router(
        streams=[
            openai_stream("", [calc("c1", "1+1"), ("c2", "write", {})]),
            openai_stream("ok"),
        ]
    )
    events = run_agent(make_agent(router, mutating_registry(calls), approver))
    assert asked == ["write"]
    assert calls == [{}]
    assert [e.tool_result.output for e in events if e.kind == "tool_result"] == ["2", "written"]


def test_interruption_keeps_history_consistent():
    async def stopper(arguments, ctx):
        raise Stop()

    registry = ToolRegistry([Tool("stopper", "d", EMPTY, stopper)])
    router, _ = scripted_router(
        streams=[openai_stream("", [("a", "stopper", {}), ("b", "stopper", {})])]
    )
    agent = make_agent(router, registry)
    with pytest.raises(Stop):
        run_agent(agent)
    tail = agent.history[-2:]
    assert [m.role for m in tail] == ["tool", "tool"]
    assert [m.tool_call_id for m in tail] == ["a", "b"]
    assert all("cancelled" in m.content for m in tail)


def test_conversation_continues_across_runs_and_reset_clears():
    router, transport = scripted_router(streams=[openai_stream("one"), openai_stream("two")])
    agent = make_agent(router)
    run_agent(agent, "first")
    run_agent(agent, "second")
    roles = [m["role"] for m in transport.calls[1].payload["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    agent.reset()
    assert [m.role for m in agent.history] == ["system"]
    assert agent.usage.total_tokens == 0


def test_tool_result_message_roundtrip():
    result = ToolResult("c1", "calculate", "14")
    assert result.to_message().tool_call_id == "c1"


def test_system_prompt_includes_environment():
    prompt = build_system_prompt(Path("/work"), "", date(2026, 1, 2))
    assert "Eurus" in prompt
    assert "/work" in prompt
    assert "2026-01-02" in prompt
    custom = build_system_prompt(Path("/work"), "Be terse.", date(2026, 1, 2))
    assert "<eurus-tier:stable>" in custom
    assert "Be terse." in custom
    assert "EURUS" not in custom


def test_mutating_tool_fails_closed_without_approver():
    calls = []
    router, _ = scripted_router(
        streams=[openai_stream("", [("c1", "write", {})]), openai_stream("ok")]
    )
    events = run_agent(make_agent(router, mutating_registry(calls)))
    result = next(event.tool_result for event in events if event.kind == "tool_result")
    assert result is not None and result.is_error
    assert "no approval provider" in result.output
    assert calls == []


def test_independent_tool_calls_execute_concurrently_and_keep_result_order():
    import time

    completion_order = []

    async def delayed(arguments, ctx):
        delay = float(arguments["delay"])
        await asyncio.sleep(delay)
        completion_order.append(arguments["label"])
        return arguments["label"]

    registry = ToolRegistry(
        [
            Tool(
                "delayed",
                "wait and return label",
                {
                    "type": "object",
                    "properties": {
                        "delay": {"type": "number"},
                        "label": {"type": "string"},
                    },
                    "required": ["delay", "label"],
                    "additionalProperties": False,
                },
                delayed,
            )
        ]
    )
    router, _ = scripted_router(
        streams=[
            openai_stream(
                "",
                [
                    ("slow", "delayed", {"delay": 0.12, "label": "slow"}),
                    ("fast", "delayed", {"delay": 0.01, "label": "fast"}),
                ],
            ),
            openai_stream("done"),
        ]
    )
    agent = make_agent(router, registry)
    started = time.monotonic()
    events = run_agent(agent)
    elapsed = time.monotonic() - started

    results = [event.tool_result.output for event in events if event.kind == "tool_result"]
    assert results == ["slow", "fast"]
    assert completion_order == ["fast", "slow"]
    assert elapsed < 0.22
    tool_messages = [message for message in agent.history if message.role == "tool"]
    assert [message.tool_call_id for message in tool_messages] == ["slow", "fast"]


def test_max_parallel_tools_must_be_positive():
    router, _ = scripted_router(streams=[openai_stream("ok")])
    with pytest.raises(ValueError, match="max_parallel_tools"):
        Agent(
            router,
            default_registry(),
            ToolContext(workspace=Path(".")),
            system_prompt="sys",
            max_parallel_tools=0,
        )

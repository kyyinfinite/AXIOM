import asyncio
from datetime import datetime
from pathlib import Path

import pytest

from eurus.errors import ToolError
from eurus.models import ToolCall
from eurus.tools import Tool, ToolContext, ToolRegistry, default_registry, truncate_text

EMPTY = {"type": "object", "properties": {}}


def context(**kwargs):
    return ToolContext(workspace=Path("."), **kwargs)


def execute(registry, name, arguments, **kwargs):
    return asyncio.run(registry.execute(ToolCall("1", name, arguments), context(**kwargs)))


async def echo(arguments, ctx):
    return "echo"


def test_register_rejects_duplicates_and_bad_names():
    registry = ToolRegistry([Tool("a", "d", EMPTY, echo)])
    with pytest.raises(ValueError):
        registry.register(Tool("a", "d", EMPTY, echo))
    with pytest.raises(ValueError):
        registry.register(Tool("bad name", "d", EMPTY, echo))


def test_decorator_registration_and_lookup():
    registry = ToolRegistry()

    @registry.tool("hello", "say hello", EMPTY)
    async def hello(arguments, ctx):
        return "hi"

    assert "hello" in registry
    assert len(registry) == 1
    assert registry.get("hello").handler is hello
    assert registry.get("missing") is None
    assert [spec.name for spec in registry.specs()] == ["hello"]


def test_specs_are_sorted():
    registry = ToolRegistry([Tool("b", "d", EMPTY, echo), Tool("a", "d", EMPTY, echo)])
    assert registry.names() == ["a", "b"]
    assert [spec.name for spec in registry.specs()] == ["a", "b"]


def test_execute_success():
    result = execute(ToolRegistry([Tool("a", "d", EMPTY, echo)]), "a", {})
    assert result.output == "echo"
    assert not result.is_error


def test_execute_unknown_tool_lists_available():
    result = execute(ToolRegistry([Tool("a", "d", EMPTY, echo)]), "nope", {})
    assert result.is_error
    assert "unknown tool 'nope'" in result.output
    assert "a" in result.output


def test_execute_validation_error():
    registry = default_registry()
    result = execute(registry, "calculate", {})
    assert result.is_error
    assert "expression is required" in result.output


def test_execute_invalid_json_arguments():
    result = execute(default_registry(), "calculate", {"_invalid_arguments": "oops"})
    assert result.is_error
    assert "not valid JSON" in result.output


def test_execute_reports_tool_error_and_crash():
    async def fail(arguments, ctx):
        raise ToolError("nope")

    async def crash(arguments, ctx):
        raise RuntimeError("kaput")

    registry = ToolRegistry([Tool("fail", "d", EMPTY, fail), Tool("crash", "d", EMPTY, crash)])
    assert execute(registry, "fail", {}).output == "error: nope"
    crashed = execute(registry, "crash", {})
    assert crashed.is_error
    assert "RuntimeError: kaput" in crashed.output


def test_execute_timeout():
    async def slow(arguments, ctx):
        await asyncio.sleep(2)
        return "late"

    result = execute(ToolRegistry([Tool("slow", "d", EMPTY, slow)]), "slow", {}, timeout=0.05)
    assert result.is_error
    assert "timed out" in result.output


def test_execute_truncates_output():
    async def big(arguments, ctx):
        return "x" * 5000

    result = execute(ToolRegistry([Tool("big", "d", EMPTY, big)]), "big", {}, max_output=1000)
    assert len(result.output) < 1100
    assert "characters truncated" in result.output


def test_truncate_text():
    assert truncate_text("short", 100) == "short"
    cut = truncate_text("a" * 100 + "b" * 100, 50)
    assert cut.startswith("a" * 35)
    assert cut.endswith("b" * 15)
    assert "150 characters truncated" in cut
    assert truncate_text("abc", 0) == "abc"


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2+3*4", "14"),
        ("10/4", "2.5"),
        ("sqrt(16)", "4"),
        ("2**10", "1024"),
        ("-(3+4)", "-7"),
        ("7//2", "3"),
        ("7%4", "3"),
        ("abs(-5)", "5"),
        ("pi*2", "6.28318530717959"),
    ],
)
def test_calculate_results(expression, expected):
    result = execute(default_registry(), "calculate", {"expression": expression})
    assert result.output == expected, result.output


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "1/0",
        "9**9**9",
        "2**5000",
        "(",
        "'a'*5",
        "True",
        "sqrt(-1)",
        "x + 1",
        "a" * 300,
    ],
)
def test_calculate_rejects_bad_expressions(expression):
    result = execute(default_registry(), "calculate", {"expression": expression})
    assert result.is_error, result.output


def test_get_time_local_and_utc_and_unknown():
    registry = default_registry()
    local = execute(registry, "get_time", {})
    assert datetime.fromisoformat(local.output)
    utc = execute(registry, "get_time", {"timezone": "UTC"})
    assert utc.output.endswith("+00:00")
    unknown = execute(registry, "get_time", {"timezone": "Mars/Base"})
    assert unknown.is_error
    assert "unknown timezone" in unknown.output

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from eurus.errors import ToolError
from eurus.tools.base import Tool, ToolContext, ToolRegistry

Number = int | float

MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 1000
MAX_RESULT_BITS = 10000

CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}
FUNCTIONS: dict[str, Callable[[Number], Number]] = {"sqrt": math.sqrt, "abs": abs}


def _power(base: Number, exponent: Number) -> Number:
    if abs(exponent) > MAX_EXPONENT:
        raise ToolError("exponent is too large")
    if (
        isinstance(base, int)
        and isinstance(exponent, int)
        and exponent > 0
        and base.bit_length() * exponent > MAX_RESULT_BITS
    ):
        raise ToolError("result is too large")
    return base**exponent


BINARY: dict[type[ast.operator], Callable[[Any, Any], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _power,
}
UNARY: dict[type[ast.unaryop], Callable[[Any], Number]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate(node: ast.AST) -> Number:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float) and not isinstance(node.value, bool):
            return node.value
    elif isinstance(node, ast.BinOp) and type(node.op) in BINARY:
        return BINARY[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    elif isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
        return UNARY[type(node.op)](_evaluate(node.operand))
    elif isinstance(node, ast.Name) and node.id in CONSTANTS:
        return CONSTANTS[node.id]
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in FUNCTIONS
        and len(node.args) == 1
        and not node.keywords
    ):
        return FUNCTIONS[node.func.id](_evaluate(node.args[0]))
    raise ToolError("unsupported expression")


def _format(value: Number) -> str:
    if isinstance(value, float):
        return format(value, ".15g")
    return str(value)


async def _calculate(arguments: dict[str, Any], context: ToolContext) -> str:
    expression = str(arguments["expression"]).strip()
    if not expression or len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolError(f"expression must be 1 to {MAX_EXPRESSION_LENGTH} characters")
    try:
        tree = ast.parse(expression, mode="eval")
        return _format(_evaluate(tree))
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ToolError(f"cannot evaluate expression: {type(exc).__name__}") from exc
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    except OverflowError as exc:
        raise ToolError("number is too large") from exc


async def _get_time(arguments: dict[str, Any], context: ToolContext) -> str:
    zone = str(arguments.get("timezone") or "").strip()
    if not zone:
        now = datetime.now().astimezone()
    elif zone.upper() == "UTC":
        now = datetime.now(UTC)
    else:
        try:
            now = datetime.now(ZoneInfo(zone))
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ToolError(f"unknown timezone '{zone}'") from exc
    return now.isoformat(timespec="seconds")


def builtin_tools() -> list[Tool]:
    return [
        Tool(
            name="calculate",
            description=(
                "Evaluate an arithmetic expression. Supports + - * / // % **, parentheses, "
                "the constants pi and e, and the functions sqrt and abs."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "For example: 2 + 3 * 4"}
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
            handler=_calculate,
        ),
        Tool(
            name="get_time",
            description="Get the current date and time, optionally in an IANA timezone.",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA name such as Asia/Jakarta, or UTC. Defaults to local.",
                    }
                },
                "additionalProperties": False,
            },
            handler=_get_time,
        ),
    ]


def default_registry(
    *,
    include_capability_tools: bool = False,
    include_workflow_tools: bool = False,
    include_phase4: bool | None = None,
    include_phase5: bool | None = None,
    memory_router: Any | None = None,
) -> ToolRegistry:

    if include_phase4 is not None:
        include_capability_tools = include_capability_tools or include_phase4
    if include_phase5 is not None:
        include_workflow_tools = include_workflow_tools or include_phase5
    registry = ToolRegistry(builtin_tools())
    if include_capability_tools:
        from eurus.agent.capability_tools import capability_tools
        from eurus.agent.web_tools import web_tools
        from eurus.coding.tools import coding_tools

        for tool in [*coding_tools(), *capability_tools(memory_router), *web_tools()]:
            registry.register(tool)
    if include_workflow_tools:
        from eurus.coding.workflow_tools import workflow_tools

        for tool in workflow_tools():
            registry.register(tool)
    return registry

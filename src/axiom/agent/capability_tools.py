from __future__ import annotations

import json
from typing import Any

from axiom.agent.planner import TaskPlanner
from axiom.memory.manager import MemoryManager
from axiom.paths import home_dir
from axiom.tools.base import Tool, ToolContext


def _memory() -> MemoryManager:
    return MemoryManager(home_dir() / "sessions.sqlite3")


async def _plan_task(arguments: dict[str, Any], context: ToolContext) -> str:
    planner = TaskPlanner(max_steps=int(arguments.get("max_steps", 8)))
    plan = planner.create_plan(str(arguments["goal"]))
    return json.dumps(
        {
            "goal": plan.goal,
            "steps": [
                {
                    "number": step.number,
                    "title": step.title,
                    "success_criteria": step.success_criteria,
                }
                for step in plan.steps
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


async def _recall_memory(arguments: dict[str, Any], context: ToolContext) -> str:
    items = _memory().recall(str(arguments["query"]), limit=int(arguments.get("limit", 5)))
    return json.dumps(items, ensure_ascii=False, indent=2) if items else "No matching memories."


async def _remember(arguments: dict[str, Any], context: ToolContext) -> str:
    memory_id = _memory().remember(
        str(arguments["content"]),
        category=str(arguments.get("category", "general")),
        tags=str(arguments.get("tags", "")),
    )
    return f"Saved memory #{memory_id}."


async def _forget(arguments: dict[str, Any], context: ToolContext) -> str:
    memory_id = int(arguments["memory_id"])
    return f"Deleted memory #{memory_id}." if _memory().forget(memory_id) else "Memory not found."


def capability_tools() -> list[Tool]:
    return [
        Tool(
            name="plan_task",
            description="Create a bounded, practical plan for a multi-step task before acting.",
            parameters={
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "max_steps": {"type": "integer"},
                },
                "required": ["goal"],
                "additionalProperties": False,
            },
            handler=_plan_task,
        ),
        Tool(
            name="recall_memory",
            description="Search the user's explicitly stored local memories by keywords.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=_recall_memory,
        ),
        Tool(
            name="remember_memory",
            description=(
                "Persist a useful user-approved fact or project decision. Requires owner approval."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "category": {"type": "string"},
                    "tags": {"type": "string"},
                },
                "required": ["content"],
                "additionalProperties": False,
            },
            handler=_remember,
            mutates=True,
            requires_approval=True,
        ),
        Tool(
            name="forget_memory",
            description="Delete a stored memory by ID. Requires owner approval.",
            parameters={
                "type": "object",
                "properties": {"memory_id": {"type": "integer"}},
                "required": ["memory_id"],
                "additionalProperties": False,
            },
            handler=_forget,
            mutates=True,
            requires_approval=True,
        ),
    ]

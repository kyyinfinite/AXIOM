import asyncio
from pathlib import Path

import pytest

from eurus.agent.delegation import (
    MAX_SUBAGENT_ITERATIONS,
    build_subagent_registry,
    delegation_tools,
    run_subagent,
)
from eurus.providers.testing import openai_message, scripted_router
from eurus.tools.base import Tool, ToolContext, ToolRegistry

EMPTY = {"type": "object", "properties": {}}


def test_subagent_registry_is_restricted_to_read_only_allowlist():
    async def handler(arguments, context):
        return "ok"

    parent = ToolRegistry(
        [
            Tool("read_file", "read", EMPTY, handler),
            Tool("write_file", "write", EMPTY, handler, mutates=True),
            Tool("approve_me", "approval", EMPTY, handler, requires_approval=True),
            Tool("shell", "shell", EMPTY, handler),
        ]
    )
    child = build_subagent_registry(parent)
    assert child.names() == ["read_file"]
    assert "delegate_task" not in child
    assert "shell" not in child


def test_delegation_tool_is_approval_gated_and_non_parallel():
    router, _ = scripted_router(json_responses=[openai_message("done")])
    tools = delegation_tools(router, ToolRegistry())
    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "delegate_task"
    assert tool.requires_approval is True
    assert tool.parallel_safe is False
    assert tool.interactive is True
    assert tool.parameters["properties"]["max_iterations"]["maximum"] == MAX_SUBAGENT_ITERATIONS


def test_subagent_runs_with_bounded_budget_and_returns_result():
    async def run():
        router, _ = scripted_router(json_responses=[openai_message("Evidence found.")])
        result = await run_subagent(
            router,
            ToolRegistry(),
            ToolContext(workspace=Path.cwd()),
            "Find evidence.",
            max_iterations=3,
        )
        assert result["result"] == "Evidence found."
        assert result["iterations_used"] == 1
        assert result["iteration_limit"] == 3
        assert result["available_tools"] == []

    asyncio.run(run())


def test_subagent_rejects_empty_or_oversized_tasks():
    async def run():
        router, _ = scripted_router()
        context = ToolContext(workspace=Path.cwd())
        with pytest.raises(ValueError, match="must not be empty"):
            await run_subagent(router, ToolRegistry(), context, "  ")
        with pytest.raises(ValueError, match="exceeds"):
            await run_subagent(router, ToolRegistry(), context, "x" * 12001)
        with pytest.raises(ValueError, match="between 1 and 50"):
            await run_subagent(router, ToolRegistry(), context, "task", max_iterations=51)

    asyncio.run(run())


def test_document_team_review_uses_isolated_subagents():
    import json

    from eurus.agent.team_tools import document_team_tools

    async def run():
        router, _ = scripted_router(
            json_responses=[
                openai_message("Proofreading findings."),
                openai_message("Engagement findings."),
                openai_message("Critical findings."),
                openai_message("PRIORITIES\n1. Clarify the opening."),
            ]
        )
        registry = ToolRegistry()
        tool = document_team_tools(router, registry)[0]
        output = await tool.handler(
            {"content": "A draft document.", "task": "Review", "include_revision": True},
            ToolContext(workspace=Path.cwd()),
        )
        payload = json.loads(output)
        assert payload["workflow"] == "parallel_document_review"
        assert set(payload["reviewers_completed"]) == {
            "proofreader",
            "engagement_editor",
            "critical_reviewer",
        }
        assert "PRIORITIES" in payload["lead_editor"]

    asyncio.run(run())

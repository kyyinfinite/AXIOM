from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from eurus.agent.team_tools import _review_team, document_team_tools
from eurus.errors import ToolError
from eurus.models import Message, Usage
from eurus.providers.base import ChatResponse
from eurus.tools.base import ToolContext


class FakeRouter:
    def __init__(self) -> None:
        self.calls: list[Sequence[Message]] = []
        self.active = 0
        self.max_active = 0

    async def chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        self.calls.append(messages)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        text = (
            "review finding"
            if "lead editor" not in messages[0].content
            else "PRIORITIES\n1. Fix\nREVISED DOCUMENT\nImproved text"
        )
        return ChatResponse(message=Message.assistant(text), finish_reason="stop", usage=Usage())


@pytest.mark.asyncio
async def test_team_runs_three_reviews_in_parallel_and_synthesizes() -> None:
    router = FakeRouter()
    result = await _review_team(
        {"content": "Example document", "task": "Polish it", "include_revision": True},
        ToolContext(workspace=Path(".")),
        router,  # type: ignore[arg-type]
    )
    assert router.max_active >= 3
    assert len(router.calls) == 4
    assert '"proofreader"' in result
    assert '"engagement_editor"' in result
    assert '"critical_reviewer"' in result
    assert "REVISED DOCUMENT" in result


def test_team_tool_factory_registers_tool() -> None:
    tools = document_team_tools(object())  # type: ignore[arg-type]
    assert tools[0].name == "document_team_review"
    assert {tool.name for tool in tools} >= {"document_team_review", "office_team_create"}


@pytest.mark.asyncio
async def test_team_rejects_empty_content() -> None:
    with pytest.raises(ToolError, match="must not be empty"):
        await _review_team({"content": "   "}, ToolContext(workspace=Path(".")), object())  # type: ignore[arg-type]

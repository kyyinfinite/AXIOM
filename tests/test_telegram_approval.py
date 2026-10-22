from __future__ import annotations

import asyncio

import pytest

from eurus.gateway.telegram.approval import (
    ApprovalContext,
    TelegramApprovalBroker,
    current_approval_context,
)
from eurus.models import ToolCall
from eurus.tools.base import Tool, ToolContext


async def _handler(arguments: dict[str, object], context: ToolContext) -> str:
    return str(arguments)


@pytest.mark.asyncio
async def test_approval_requires_matching_owner_and_chat() -> None:
    broker = TelegramApprovalBroker(timeout=1.0)
    messages: list[tuple[int, str]] = []

    async def notify(chat_id: int, text: str) -> None:
        messages.append((chat_id, text))

    broker.bind_notifier(notify)
    context_token = current_approval_context.set(ApprovalContext("task-1", "42", 100))
    tool = Tool("write_file", "Write a file", {}, _handler, mutates=True)
    approval = asyncio.create_task(
        broker.approve(ToolCall("call-1", "write_file", {"path": "a.txt"}), tool)
    )
    try:
        for _ in range(20):
            if messages:
                break
            await asyncio.sleep(0)
        token = messages[0][1].split("/approve ", 1)[1].splitlines()[0]
        assert await broker.resolve(token, owner_id="7", chat_id=100, allowed=True) is False
        assert await broker.resolve(token, owner_id="42", chat_id=101, allowed=True) is False
        assert await broker.resolve(token, owner_id="42", chat_id=100, allowed=True) is True
        assert await approval is True
        assert await broker.resolve(token, owner_id="42", chat_id=100, allowed=True) is False
    finally:
        current_approval_context.reset(context_token)
        await broker.close()


@pytest.mark.asyncio
async def test_approval_fails_closed_without_task_context() -> None:
    broker = TelegramApprovalBroker(timeout=0.1)
    tool = Tool("write_file", "Write a file", {}, _handler, mutates=True)
    assert await broker.approve(ToolCall("call-1", "write_file"), tool) is False
    await broker.close()


@pytest.mark.asyncio
async def test_approval_timeout_denies_action() -> None:
    broker = TelegramApprovalBroker(timeout=0.01)

    async def notify(chat_id: int, text: str) -> None:
        return None

    broker.bind_notifier(notify)
    context_token = current_approval_context.set(ApprovalContext("task-2", "42", 100))
    tool = Tool("write_file", "Write a file", {}, _handler, mutates=True)
    try:
        result = await broker.approve(ToolCall("call-2", "write_file"), tool)
        assert result is False
        assert broker.pending == {}
    finally:
        current_approval_context.reset(context_token)
        await broker.close()

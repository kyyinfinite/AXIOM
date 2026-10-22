from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from eurus.gateway.tasks import TaskStore
from eurus.gateway.telegram.adapter import BackgroundTaskManager


@pytest.mark.asyncio
async def test_cancelling_task_waiting_for_capacity_updates_persistent_status(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def runner(task: object) -> str:
        started.set()
        await release.wait()
        return "done"

    async def notify(chat_id: int, text: str) -> None:
        return None

    store = TaskStore(tmp_path / "tasks.sqlite3")
    manager = BackgroundTaskManager(store, runner, notify, max_concurrent=1)
    first = await manager.submit("42", "session-a", "first", notify_chat_id=100)
    await started.wait()
    second = await manager.submit("42", "session-a", "second", notify_chat_id=100)

    assert await manager.cancel(second.task_id, "42") is True
    await asyncio.sleep(0)
    updated = store.get(second.task_id, owner_id="42")
    assert updated is not None
    assert updated.status == "cancelled"

    assert await manager.cancel(first.task_id, "42") is True
    await manager.shutdown()


@pytest.mark.asyncio
async def test_event_bridge_forwards_tool_progress_and_unsubscribes(tmp_path: Path) -> None:
    from eurus.agent.events import AgentEvent
    from eurus.models import ToolCall
    from eurus.runtime.event_bridge import AgentEventBridge

    bridge = AgentEventBridge()
    notifications: list[str] = []

    async def runner(task: object) -> str:
        await bridge.publish("session-a", AgentEvent.tool_started(ToolCall("call-1", "read_file")))
        return "finished"

    async def notify(chat_id: int, text: str) -> None:
        notifications.append(text)

    manager = BackgroundTaskManager(
        TaskStore(tmp_path / "tasks.sqlite3"),
        runner,
        notify,
        max_concurrent=1,
        event_bridge=bridge,
    )
    task = await manager.submit("42", "session-a", "inspect", notify_chat_id=100)
    await manager.active[task.task_id]

    assert "Working with tool: read_file" in notifications
    assert notifications[-1] == "finished"
    assert bridge._subscribers == {}
    await manager.shutdown()
    bridge.close()


@pytest.mark.asyncio
async def test_shutdown_pauses_task_even_if_worker_has_not_started(tmp_path: Path) -> None:
    async def runner(task: object) -> str:
        await asyncio.Event().wait()
        return "unreachable"

    async def notify(chat_id: int, text: str) -> None:
        return None

    store = TaskStore(tmp_path / "tasks.sqlite3")
    manager = BackgroundTaskManager(store, runner, notify, max_concurrent=1)
    task = await manager.submit("42", "session-a", "long work", notify_chat_id=100)

    await manager.shutdown()

    updated = store.get(task.task_id, owner_id="42")
    assert updated is not None
    assert updated.status == "paused"
    assert updated.error == "gateway shutdown; review before resuming"

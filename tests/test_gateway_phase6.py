from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from eurus.gateway.security import GatewayAccessPolicy, GatewayIdentity, parse_id_allowlist
from eurus.gateway.tasks import TaskStore
from eurus.gateway.telegram import BackgroundTaskManager


def test_numeric_allowlist_and_private_default() -> None:
    assert parse_id_allowlist("123, 456,123") == {123, 456}
    assert parse_id_allowlist("-100123", allow_negative=True) == {-100123}
    with pytest.raises(ValueError):
        parse_id_allowlist("@username")
    policy = GatewayAccessPolicy({123}, {-100123})
    assert policy.permits(GatewayIdentity(123, 123))
    assert not policy.permits(GatewayIdentity(456, 456))
    assert not policy.permits(GatewayIdentity(123, -100999, is_group=True))
    assert policy.permits(GatewayIdentity(123, -100123, is_group=True))


def test_task_store_transitions_ownership_and_update_dedup(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create("100", "telegram:100:100", "test objective")
    assert task.status == "queued"
    assert store.get(task.task_id, owner_id="200") is None
    assert store.claim_update("telegram", "77")
    assert not store.claim_update("telegram", "77")
    running = store.transition(task.task_id, "running", owner_id="100")
    assert running.status == "running"
    with pytest.raises(ValueError):
        store.transition(task.task_id, "queued")
    completed = store.transition(task.task_id, "completed", state={"result": "ok"})
    assert completed.state == {"result": "ok"}
    with pytest.raises(ValueError):
        store.transition(task.task_id, "running")


def test_task_state_persists_across_store_restart(tmp_path: Path) -> None:
    path = tmp_path / "tasks.sqlite3"
    task = TaskStore(path).create("42", "telegram:42:42", "durable objective", {"checkpoint": 3})
    restored = TaskStore(path).get(task.task_id)
    assert restored is not None
    assert restored.state == {"checkpoint": 3}
    assert restored.objective == "durable objective"


@pytest.mark.asyncio
async def test_background_manager_runs_and_persists_result(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    notifications: list[tuple[int, str]] = []

    async def runner(task):
        await asyncio.sleep(0)
        return "verified output"

    async def notify(chat_id: int, text: str) -> None:
        notifications.append((chat_id, text))

    manager = BackgroundTaskManager(store, runner, notify, max_concurrent=1)
    task = await manager.submit("7", "telegram:7:70", "do work", notify_chat_id=70)
    worker = manager.active[task.task_id]
    await worker
    result = store.get(task.task_id, owner_id="7")
    assert result is not None and result.status == "completed"
    assert result.state["result"] == "verified output"
    assert notifications and notifications[0][0] == 70


@pytest.mark.asyncio
async def test_background_manager_failure_is_recorded(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")

    async def runner(task):
        raise TimeoutError("private detail should not be exposed")

    async def notify(chat_id: int, text: str) -> None:
        assert "private detail" not in text

    manager = BackgroundTaskManager(store, runner, notify)
    task = await manager.submit("7", "telegram:7:70", "do work", notify_chat_id=70)
    await manager.active[task.task_id]
    result = store.get(task.task_id)
    assert result is not None and result.status == "failed"
    assert result.error == "TimeoutError"


def test_chat_lock_persists_across_store_restart(tmp_path: Path) -> None:
    path = tmp_path / "tasks.sqlite3"
    store = TaskStore(path)
    store.set_chat_locked("123", "-100456", True)
    assert TaskStore(path).is_chat_locked("123", "-100456")
    TaskStore(path).set_chat_locked("123", "-100456", False)
    assert not TaskStore(path).is_chat_locked("123", "-100456")


def test_new_session_preserves_previous_session_identity(tmp_path: Path) -> None:
    path = tmp_path / "tasks.sqlite3"
    store = TaskStore(path)
    first = store.current_session("123", "456")
    second = store.reset_session("123", "456")
    assert first == "telegram:123:456"
    assert second != first
    assert TaskStore(path).current_session("123", "456") == second


@pytest.mark.asyncio
async def test_background_manager_enforces_task_timeout(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")

    async def runner(task):
        await asyncio.sleep(0.05)
        return "too late"

    async def notify(chat_id: int, text: str) -> None:
        assert "failed" in text

    manager = BackgroundTaskManager(store, runner, notify, task_timeout=0.001)
    task = await manager.submit("7", "telegram:7:70", "do work", notify_chat_id=70)
    await manager.active[task.task_id]
    result = store.get(task.task_id)
    assert result is not None and result.status == "failed"
    assert result.error == "TimeoutError"

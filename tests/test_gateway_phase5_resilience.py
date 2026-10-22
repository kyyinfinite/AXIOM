from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from eurus.gateway.contracts import GatewayAdapter, GatewayMessage
from eurus.gateway.tasks import TaskStore
from eurus.gateway.telegram.adapter import BackgroundTaskManager, split_telegram_text


def test_split_telegram_text_preserves_content_and_respects_limit() -> None:
    text = ("alpha beta gamma\n" * 1200).strip()
    chunks = split_telegram_text(text, limit=128)

    assert len(chunks) > 1
    assert all(len(chunk) <= 128 for chunk in chunks)
    assert "".join(chunks) == text


def test_split_telegram_text_validates_limit() -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        split_telegram_text("hello", 0)


def test_gateway_contract_accepts_async_adapter() -> None:
    class FakeAdapter:
        @property
        def platform(self) -> str:
            return "test"

        async def send_text(self, conversation_id: str, text: str) -> None:
            return None

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    adapter: GatewayAdapter = FakeAdapter()
    message = GatewayMessage("test", "user-1", "chat-1", "message-1", "hello")

    assert adapter.platform == message.platform
    assert message.text == "hello"


@pytest.mark.asyncio
async def test_long_task_result_is_passed_to_delivery_without_truncation(
    tmp_path: Path,
) -> None:
    result = "result-line\n" * 1200
    delivered: list[str] = []

    async def runner(task: object) -> str:
        return result

    async def notify(chat_id: int, text: str) -> None:
        delivered.extend(split_telegram_text(text, limit=256))

    manager = BackgroundTaskManager(TaskStore(tmp_path / "tasks.sqlite3"), runner, notify)
    task = await manager.submit("42", "session-a", "long result", notify_chat_id=100)
    await manager.active[task.task_id]

    assert "".join(delivered) == result
    assert all(len(chunk) <= 256 for chunk in delivered)
    await manager.shutdown()


@pytest.mark.asyncio
async def test_telegram_notifier_retries_rate_limits_and_preserves_text() -> None:
    from eurus.gateway.telegram.adapter import TelegramNotifier

    attempts = 0
    delivered: list[tuple[int, str]] = []
    delays: list[float] = []

    class RateLimitError(Exception):
        retry_after = 0.25

    async def sender(chat_id: int, text: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RateLimitError()
        delivered.append((chat_id, text))

    async def sleep(delay: float) -> None:
        delays.append(delay)

    notifier = TelegramNotifier(
        sender,
        global_interval=0,
        chat_interval=0,
        sleep=sleep,
    )
    await notifier(100, "hello")

    assert attempts == 2
    assert delivered == [(100, "hello")]
    assert delays == [0.25]


@pytest.mark.asyncio
async def test_telegram_notifier_uses_per_chat_and_global_rate_limits() -> None:
    from eurus.gateway.telegram.adapter import TelegramNotifier

    current_time = 0.0
    delays: list[float] = []
    delivered: list[tuple[int, str]] = []

    async def sender(chat_id: int, text: str) -> None:
        delivered.append((chat_id, text))

    async def sleep(delay: float) -> None:
        nonlocal current_time
        delays.append(delay)
        current_time += delay

    notifier = TelegramNotifier(
        sender,
        global_interval=0.1,
        chat_interval=1.0,
        sleep=sleep,
        clock=lambda: current_time,
    )
    await notifier(100, "first")
    await notifier(200, "second")
    await notifier(100, "third")

    assert delivered == [(100, "first"), (200, "second"), (100, "third")]
    assert sum(delays) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_telegram_notifier_does_not_retry_permanent_errors() -> None:
    from eurus.gateway.telegram.adapter import TelegramDeliveryError, TelegramNotifier

    attempts = 0

    async def sender(chat_id: int, text: str) -> None:
        nonlocal attempts
        attempts += 1
        raise ValueError("invalid chat")

    notifier = TelegramNotifier(sender, global_interval=0, chat_interval=0)

    with pytest.raises(TelegramDeliveryError, match="delivery failed"):
        await notifier(100, "hello")

    assert attempts == 1


@pytest.mark.asyncio
async def test_notification_failure_does_not_rewrite_completed_task_as_failed(
    tmp_path: Path,
) -> None:
    async def runner(task: object) -> str:
        return "finished"

    async def notify(chat_id: int, text: str) -> None:
        raise OSError("network unavailable")

    store = TaskStore(tmp_path / "tasks.sqlite3")
    manager = BackgroundTaskManager(store, runner, notify)
    task = await manager.submit("42", "session-a", "complete task", notify_chat_id=100)
    await manager.active[task.task_id]

    updated = store.get(task.task_id, owner_id="42")
    assert updated is not None
    assert updated.status == "completed"
    assert updated.state["result"] == "finished"
    await manager.shutdown()


@pytest.mark.asyncio
async def test_concurrent_submissions_cannot_exceed_pending_capacity(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def runner(task: object) -> str:
        started.set()
        await release.wait()
        return "finished"

    async def notify(chat_id: int, text: str) -> None:
        return None

    store = TaskStore(tmp_path / "tasks.sqlite3")
    manager = BackgroundTaskManager(store, runner, notify, max_concurrent=1)
    outcomes = await asyncio.gather(
        *(
            manager.submit("42", "session-a", f"task {index}", notify_chat_id=100)
            for index in range(manager.max_pending + 2)
        ),
        return_exceptions=True,
    )

    accepted = [item for item in outcomes if not isinstance(item, BaseException)]
    rejected = [item for item in outcomes if isinstance(item, RuntimeError)]
    assert len(accepted) == manager.max_pending
    assert len(rejected) == 2
    await started.wait()
    release.set()
    await manager.shutdown()


@pytest.mark.parametrize(
    "interrupted_status",
    ["queued", "running", "waiting_approval", "waiting_input", "recovering"],
)
def test_interrupted_tasks_are_paused_for_manual_review(
    tmp_path: Path,
    interrupted_status: str,
) -> None:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create("42", "session-a", "possibly side-effectful operation")
    if interrupted_status != "queued":
        store.transition(task.task_id, "running")
    if interrupted_status in {"waiting_approval", "waiting_input", "recovering"}:
        store.transition(task.task_id, interrupted_status)

    assert store.pause_interrupted() == 1
    updated = store.get(task.task_id, owner_id="42")
    assert updated is not None
    assert updated.status == "paused"
    assert "review before resubmitting" in updated.error

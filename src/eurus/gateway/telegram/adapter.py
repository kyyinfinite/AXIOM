from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from eurus.gateway.security import GatewayAccessPolicy, GatewayIdentity
from eurus.gateway.tasks import DurableTask, TaskStore
from eurus.gateway.telegram.approval import (
    ApprovalContext,
    TelegramApprovalBroker,
    current_approval_context,
)
from eurus.runtime.event_bridge import AgentEventBridge, RuntimeEvent

log = logging.getLogger(__name__)
MAX_TELEGRAM_TEXT = 4000


@dataclass(frozen=True, slots=True)
class TelegramMessage:
    user_id: int
    chat_id: int
    message_id: int
    update_id: int
    text: str
    is_group: bool = False


TaskRunner = Callable[[DurableTask], Awaitable[str]]
Notifier = Callable[[int, str], Awaitable[None]]
MessageSender = Callable[[int, str], Awaitable[object]]


class TelegramDeliveryError(RuntimeError):
    pass


class TelegramNotifier:
    def __init__(
        self,
        sender: MessageSender,
        *,
        max_attempts: int = 3,
        global_interval: float = 1 / 30,
        chat_interval: float = 1.0,
        group_interval: float = 3.0,
        max_retry_after: float = 120.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if global_interval < 0 or chat_interval < 0 or group_interval < 0:
            raise ValueError("rate-limit intervals cannot be negative")
        if max_retry_after < 0:
            raise ValueError("max_retry_after cannot be negative")
        self.sender = sender
        self.max_attempts = max_attempts
        self.global_interval = global_interval
        self.chat_interval = chat_interval
        self.group_interval = group_interval
        self.max_retry_after = max_retry_after
        self.sleep = sleep
        self.clock = clock
        self._schedule_lock = asyncio.Lock()
        self._next_global_send = 0.0
        self._next_chat_send: dict[int, float] = {}

    async def __call__(self, chat_id: int, text: str) -> None:
        for chunk in split_telegram_text(text):
            await self._send_chunk(chat_id, chunk)

    async def _reserve_send_slot(self, chat_id: int) -> None:
        async with self._schedule_lock:
            now = self.clock()
            next_chat = self._next_chat_send.get(chat_id, now)
            chat_interval = self.group_interval if chat_id < 0 else self.chat_interval
            scheduled = max(now, self._next_global_send, next_chat)
            self._next_global_send = scheduled + self.global_interval
            self._next_chat_send[chat_id] = scheduled + chat_interval
            if len(self._next_chat_send) > 2048:
                self._next_chat_send = {
                    key: value
                    for key, value in self._next_chat_send.items()
                    if value > now
                }
                self._next_chat_send[chat_id] = scheduled + chat_interval
        delay = scheduled - now
        if delay > 0:
            await self.sleep(delay)

    async def _send_chunk(self, chat_id: int, text: str) -> None:
        for attempt in range(self.max_attempts):
            await self._reserve_send_slot(chat_id)
            try:
                await self.sender(chat_id, text)
                return
            except Exception as exc:
                retry_after = getattr(exc, "retry_after", None)
                error_name = type(exc).__name__
                retryable = retry_after is not None or error_name in {
                    "TelegramNetworkError",
                    "TelegramServerError",
                    "TimeoutError",
                    "ConnectionError",
                    "OSError",
                }
                if not retryable:
                    raise TelegramDeliveryError("Telegram message delivery failed") from exc
                if attempt + 1 >= self.max_attempts:
                    raise TelegramDeliveryError(
                        "Telegram message delivery retries exhausted"
                    ) from exc
                if retry_after is not None:
                    try:
                        delay = max(0.0, float(retry_after))
                    except (TypeError, ValueError) as conversion_error:
                        raise TelegramDeliveryError(
                            "Telegram returned an invalid retry delay"
                        ) from conversion_error
                    if delay > self.max_retry_after:
                        raise TelegramDeliveryError(
                            "Telegram retry delay exceeds the configured limit"
                        ) from exc
                else:
                    delay = min(0.5 * (2**attempt), self.max_retry_after)
                await self.sleep(delay)


class BackgroundTaskManager:
    def __init__(
        self,
        store: TaskStore,
        runner: TaskRunner,
        notifier: Notifier,
        *,
        max_concurrent: int = 2,
        task_timeout: float = 3600.0,
        event_bridge: AgentEventBridge | None = None,
    ) -> None:
        if not 1 <= max_concurrent <= 8:
            raise ValueError("max_concurrent must be between 1 and 8")
        if task_timeout <= 0:
            raise ValueError("task_timeout must be positive")
        self.store = store
        self.runner = runner
        self.task_timeout = task_timeout
        self.notifier = notifier
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self.max_pending = max_concurrent * 8
        self.active: dict[str, asyncio.Task[None]] = {}
        self.cancelled: set[str] = set()
        self._submit_lock = asyncio.Lock()
        self.approval_broker: TelegramApprovalBroker | None = None
        self.event_bridge = event_bridge

    async def submit(
        self,
        owner_id: str,
        session_id: str,
        objective: str,
        *,
        notify_chat_id: int | None = None,
    ) -> DurableTask:
        async with self._submit_lock:
            active_count = sum(not worker.done() for worker in self.active.values())
            if active_count >= self.max_pending:
                raise RuntimeError("background task capacity reached")
            state = {
                "notify_chat_id": (
                    notify_chat_id if notify_chat_id is not None else int(owner_id)
                )
            }
            task = self.store.create(owner_id, session_id, objective, state)
            worker = asyncio.create_task(self._execute(task), name=f"eurus-task-{task.task_id}")
            self.active[task.task_id] = worker
            worker.add_done_callback(lambda _: self.active.pop(task.task_id, None))
            return task

    async def _execute(self, task: DurableTask) -> None:
        context_token: contextvars.Token[ApprovalContext | None] | None = None
        subscriber_id: str | None = None
        event_queue: asyncio.Queue[RuntimeEvent] | None = None
        event_worker: asyncio.Task[None] | None = None
        try:
            async with self.semaphore:
                current = self.store.get(task.task_id)
                if current is None or current.status != "queued":
                    return
                self.store.transition(task.task_id, "running")
                chat_id = int(task.state.get("notify_chat_id", task.owner_id))
                context_token = current_approval_context.set(
                    ApprovalContext(
                        task_id=task.task_id,
                        owner_id=task.owner_id,
                        chat_id=chat_id,
                    )
                )
                if self.event_bridge is not None:
                    subscriber_id, event_queue = self.event_bridge.subscribe(
                        task.session_id, capacity=32
                    )
                    event_worker = asyncio.create_task(
                        self._forward_progress(chat_id, event_queue),
                        name=f"eurus-events-{task.task_id}",
                    )
                try:
                    result = await asyncio.wait_for(self.runner(task), timeout=self.task_timeout)
                    if task.task_id in self.cancelled:
                        current = self.store.get(task.task_id)
                        if current and current.status in {"running", "waiting_approval"}:
                            self.store.transition(task.task_id, "cancelled")
                        await self._notify_safely(
                            chat_id, "Your request was cancelled.", task.task_id
                        )
                        return
                    if event_queue is not None:
                        with contextlib.suppress(TimeoutError):
                            await asyncio.wait_for(event_queue.join(), timeout=0.5)
                    self.store.transition(task.task_id, "completed", state={"result": result})
                    await self._notify_safely(chat_id, result, task.task_id)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    current = self.store.get(task.task_id)
                    if current and current.status in {
                        "running",
                        "waiting_approval",
                        "waiting_input",
                        "recovering",
                    }:
                        self.store.transition(task.task_id, "failed", error=type(exc).__name__)
                    log.warning(
                        "Background task failed task_id=%s error_type=%s",
                        task.task_id,
                        type(exc).__name__,
                    )
                    await self._notify_safely(
                        chat_id, "Request failed. Please try again.", task.task_id
                    )
        except asyncio.CancelledError:
            if self.approval_broker is not None:
                await self.approval_broker.cancel_task(task.task_id)
            current = self.store.get(task.task_id)
            if current and current.status in {
                "queued",
                "running",
                "waiting_approval",
                "waiting_input",
                "recovering",
            }:
                self.store.transition(task.task_id, "cancelled")
            raise
        finally:
            if event_worker is not None:
                event_worker.cancel()
                await asyncio.gather(event_worker, return_exceptions=True)
            if subscriber_id is not None and self.event_bridge is not None:
                self.event_bridge.unsubscribe(task.session_id, subscriber_id)
            if context_token is not None:
                current_approval_context.reset(context_token)

    async def _notify_safely(self, chat_id: int, text: str, task_id: str) -> None:
        try:
            await self.notifier(chat_id, text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning(
                "Task notification failed task_id=%s error_type=%s",
                task_id,
                type(exc).__name__,
            )

    async def _forward_progress(self, chat_id: int, events: asyncio.Queue[RuntimeEvent]) -> None:
        while True:
            envelope = await events.get()
            try:
                event = envelope.event
                if event.kind == "tool_call" and event.tool_call is not None:
                    tool_name = event.tool_call.name[:80]
                    try:
                        await self.notifier(chat_id, f"Working with tool: {tool_name}")
                    except Exception as exc:
                        log.debug(
                            "Telegram progress update failed error_type=%s",
                            type(exc).__name__,
                        )
            finally:
                events.task_done()

    async def cancel(self, task_id: str, owner_id: str) -> bool:
        task = self.store.get(task_id, owner_id=owner_id)
        if task is None or task.status in {"completed", "failed", "cancelled"}:
            return False
        self.cancelled.add(task_id)
        if self.approval_broker is not None:
            await self.approval_broker.cancel_task(task_id)
        worker = self.active.get(task_id)
        current = self.store.get(task_id, owner_id=owner_id)
        if current is not None and current.status not in {
            "completed",
            "failed",
            "cancelled",
        }:
            self.store.transition(task_id, "cancelled", owner_id=owner_id)
        if worker is not None:
            worker.cancel()
        return True

    async def shutdown(self) -> None:
        workers = list(self.active.items())
        for task_id, worker in workers:
            if self.approval_broker is not None:
                await self.approval_broker.cancel_task(task_id)
            current = self.store.get(task_id)
            if current and current.status in {
                "queued",
                "running",
                "waiting_approval",
                "waiting_input",
                "recovering",
            }:
                self.store.transition(
                    task_id,
                    "paused",
                    error="gateway shutdown; review before resuming",
                )
            worker.cancel()
        if workers:
            await asyncio.gather(*(worker for _, worker in workers), return_exceptions=True)


def split_telegram_text(text: str, limit: int = MAX_TELEGRAM_TEXT) -> list[str]:
    if limit < 1:
        raise ValueError("limit must be positive")
    if not text:
        return [""]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        boundary = remaining.rfind("\n", 0, limit)
        if boundary < limit // 2:
            boundary = remaining.rfind(" ", 0, limit)
        if boundary < limit // 2:
            boundary = limit
        else:
            boundary += 1
        chunks.append(remaining[:boundary])
        remaining = remaining[boundary:]
    if remaining or not chunks:
        chunks.append(remaining)
    return chunks


async def run_telegram_polling(
    *,
    token: str,
    policy: GatewayAccessPolicy,
    store: TaskStore,
    runner: TaskRunner,
    max_concurrent: int = 2,
    approval_broker: TelegramApprovalBroker | None = None,
    event_bridge: AgentEventBridge | None = None,
) -> None:
    try:
        from aiogram import Bot, Dispatcher, F
        from aiogram.filters import Command
        from aiogram.types import Message
    except ImportError as exc:
        raise RuntimeError(
            "Telegram gateway requires the optional dependency: pip install 'eurus-agent[telegram]'"
        ) from exc

    bot = Bot(token=token)
    dispatcher = Dispatcher()
    manager: BackgroundTaskManager

    notifier = TelegramNotifier(
        lambda chat_id, text: bot.send_message(chat_id=chat_id, text=text)
    )

    manager = BackgroundTaskManager(
        store,
        runner,
        notifier,
        max_concurrent=max_concurrent,
        event_bridge=event_bridge,
    )
    broker = approval_broker or TelegramApprovalBroker()
    broker.bind_notifier(notifier)

    async def approval_state(task_id: str, waiting: bool) -> None:
        current = store.get(task_id)
        if current is None:
            return
        if waiting and current.status == "running":
            store.transition(task_id, "waiting_approval")
        elif not waiting and current.status == "waiting_approval":
            store.transition(task_id, "running")

    broker.state_callback = approval_state
    manager.approval_broker = broker

    def identity(message: Message) -> GatewayIdentity | None:
        if message.from_user is None or message.chat is None:
            return None
        return GatewayIdentity(
            user_id=message.from_user.id,
            chat_id=message.chat.id,
            is_group=message.chat.type in {"group", "supergroup"},
        )

    @dispatcher.message(Command("start", "help", "new", "status", "cancel", "lock", "unlock"))
    async def command_handler(message: Message) -> None:
        ident = identity(message)
        if ident is None or not policy.permits(ident):
            await message.answer(
                "Access denied. Ask the Eurus administrator to configure your numeric user ID."
            )
            return
        command = (message.text or "").split()[0].split("@", 1)[0].lower()
        user_id = str(ident.user_id)
        if command in {"/start", "/help"}:
            await message.answer(
                "Eurus Telegram gateway\n/new — start a fresh session\n/status — recent tasks\n"
                "/cancel TASK_ID — cancel your task\n/lock and /unlock — lock your chat"
            )
        elif command == "/new":
            store.reset_session(user_id, str(ident.chat_id))
            await message.answer("Started a fresh session. Earlier session history is preserved.")
        elif command == "/lock":
            store.set_chat_locked(user_id, str(ident.chat_id), True)
            await message.answer("Chat locked. Use /unlock to resume.")
        elif command == "/unlock":
            store.set_chat_locked(user_id, str(ident.chat_id), False)
            await message.answer("Chat unlocked.")
        elif command == "/status":
            items = store.list_owner(user_id, 10)
            lines = [
                f"{item.task_id[:8]} · {item.status} · {item.objective[:90]}" for item in items
            ]
            await message.answer("\n".join(lines) or "No tasks yet.")
        elif command == "/cancel":
            parts = (message.text or "").split(maxsplit=1)
            if len(parts) != 2:
                await message.answer("Usage: /cancel TASK_ID")
                return
            candidate = next(
                (
                    item.task_id
                    for item in store.list_owner(user_id, 100)
                    if item.task_id.startswith(parts[1])
                ),
                None,
            )
            if candidate and await manager.cancel(candidate, user_id):
                await message.answer("Cancellation requested.")
            else:
                await message.answer("Task not found or cannot be cancelled.")

    @dispatcher.message(Command("approve", "deny"))
    async def approval_handler(message: Message) -> None:
        ident = identity(message)
        if ident is None or not policy.permits(ident):
            await message.answer("Access denied.")
            return
        parts = (message.text or "").split(maxsplit=1)
        if len(parts) != 2:
            await message.answer("Usage: /approve TOKEN or /deny TOKEN")
            return
        command = parts[0].split("@", 1)[0].lower()
        resolved = await broker.resolve(
            parts[1].strip(),
            owner_id=str(ident.user_id),
            chat_id=ident.chat_id,
            allowed=command == "/approve",
        )
        if resolved:
            response = "Approval accepted." if command == "/approve" else "Action denied."
            await message.answer(response)
        else:
            await message.answer("Approval token is invalid, expired, or belongs to another chat.")

    @dispatcher.message(F.text)
    async def text_handler(message: Message) -> None:
        ident = identity(message)
        if ident is None or not policy.permits(ident):
            await message.answer("Access denied.")
            return
        text = (message.text or "").strip()
        if not text or text.startswith("/"):
            return
        if store.is_chat_locked(str(ident.user_id), str(ident.chat_id)):
            await message.answer("Chat is locked. Use /unlock first.")
            return
        update_id = str(message.message_id)
        if not store.claim_update("telegram", f"{ident.chat_id}:{update_id}"):
            return
        session_id = store.current_session(str(ident.user_id), str(ident.chat_id))
        try:
            await manager.submit(str(ident.user_id), session_id, text, notify_chat_id=ident.chat_id)
        except RuntimeError:
            await message.answer("Eurus is at its background task capacity. Try again shortly.")
            return
        await message.answer("Eurus is working on your request.")

    store.pause_interrupted()
    try:
        await dispatcher.start_polling(bot, allowed_updates=dispatcher.resolve_used_update_types())
    finally:
        await manager.shutdown()
        await broker.close()
        await bot.session.close()


def token_from_environment() -> str:
    token = os.environ.get("EURUS_TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("EURUS_TELEGRAM_BOT_TOKEN is required")
    return token

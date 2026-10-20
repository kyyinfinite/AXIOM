from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from eurus.gateway.security import GatewayAccessPolicy, GatewayIdentity
from eurus.gateway.tasks import DurableTask, TaskStore

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


class BackgroundTaskManager:
    def __init__(
        self,
        store: TaskStore,
        runner: TaskRunner,
        notifier: Notifier,
        *,
        max_concurrent: int = 2,
        task_timeout: float = 3600.0,
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

    async def submit(
        self,
        owner_id: str,
        session_id: str,
        objective: str,
        *,
        notify_chat_id: int | None = None,
    ) -> DurableTask:
        active_count = sum(not task.done() for task in self.active.values())
        if active_count >= self.max_pending:
            raise RuntimeError("background task capacity reached")
        state = {"notify_chat_id": notify_chat_id if notify_chat_id is not None else int(owner_id)}
        task = self.store.create(owner_id, session_id, objective, state)
        worker = asyncio.create_task(self._execute(task), name=f"eurus-task-{task.task_id}")
        self.active[task.task_id] = worker
        worker.add_done_callback(lambda _: self.active.pop(task.task_id, None))
        return task

    async def _execute(self, task: DurableTask) -> None:
        async with self.semaphore:
            current = self.store.get(task.task_id)
            if current is None or current.status != "queued":
                return
            self.store.transition(task.task_id, "running")
            try:
                result = await asyncio.wait_for(self.runner(task), timeout=self.task_timeout)
                if task.task_id in self.cancelled:
                    self.store.transition(task.task_id, "cancelled")
                    chat_id = int(task.state.get("notify_chat_id", task.owner_id))
                    await self.notifier(chat_id, f"Task {task.task_id[:8]} cancelled.")
                    return
                self.store.transition(task.task_id, "completed", state={"result": result})
                chat_id = int(task.state.get("notify_chat_id", task.owner_id))
                message = _chunk(f"Task {task.task_id[:8]} completed.\n{result}")
                await self.notifier(chat_id, message)
            except asyncio.CancelledError:
                current = self.store.get(task.task_id)
                if current and current.status in {"queued", "running", "recovering"}:
                    self.store.transition(task.task_id, "cancelled")
                raise
            except Exception as exc:
                current = self.store.get(task.task_id)
                if current and current.status in {"running", "recovering"}:
                    self.store.transition(task.task_id, "failed", error=type(exc).__name__)
                log.warning(
                    "Background task failed task_id=%s error_type=%s",
                    task.task_id,
                    type(exc).__name__,
                )
                chat_id = int(task.state.get("notify_chat_id", task.owner_id))
                await self.notifier(
                    chat_id, f"Task {task.task_id[:8]} failed. Check Eurus logs for details."
                )

    async def cancel(self, task_id: str, owner_id: str) -> bool:
        task = self.store.get(task_id, owner_id=owner_id)
        if task is None or task.status in {"completed", "failed", "cancelled"}:
            return False
        self.cancelled.add(task_id)
        worker = self.active.get(task_id)
        if worker is not None:
            worker.cancel()
        else:
            self.store.transition(task_id, "cancelled", owner_id=owner_id)
        return True

    async def shutdown(self) -> None:
        workers = list(self.active.values())
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)


def _chunk(text: str) -> str:
    if len(text) <= MAX_TELEGRAM_TEXT:
        return text
    return text[: MAX_TELEGRAM_TEXT - 32] + "\n[response truncated]"


async def run_telegram_polling(
    *,
    token: str,
    policy: GatewayAccessPolicy,
    store: TaskStore,
    runner: TaskRunner,
    max_concurrent: int = 2,
) -> None:
    try:
        from aiogram import Bot, Dispatcher, F
        from aiogram.exceptions import TelegramNetworkError, TelegramRetryAfter
        from aiogram.filters import Command
        from aiogram.types import Message
    except ImportError as exc:
        raise RuntimeError(
            "Telegram gateway requires the optional dependency: pip install 'eurus-agent[telegram]'"
        ) from exc

    bot = Bot(token=token)
    dispatcher = Dispatcher()
    manager: BackgroundTaskManager

    async def notify(chat_id: int, text: str) -> None:
        try:
            await bot.send_message(chat_id, _chunk(text))
        except TelegramRetryAfter as exc:
            await asyncio.sleep(min(float(exc.retry_after), 30.0))
            await bot.send_message(chat_id, _chunk(text))
        except TelegramNetworkError:
            log.warning("Telegram notification failed chat_id=%s", chat_id)

    manager = BackgroundTaskManager(store, runner, notify, max_concurrent=max_concurrent)

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
            task = await manager.submit(
                str(ident.user_id), session_id, text, notify_chat_id=ident.chat_id
            )
        except RuntimeError:
            await message.answer("Eurus is at its background task capacity. Try again shortly.")
            return
        await message.answer(f"Task queued: {task.task_id[:8]}\nUse /status to check progress.")

    store.pause_interrupted()
    try:
        await dispatcher.start_polling(bot, allowed_updates=dispatcher.resolve_used_update_types())
    finally:
        await manager.shutdown()
        await bot.session.close()


def token_from_environment() -> str:
    token = os.environ.get("EURUS_TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("EURUS_TELEGRAM_BOT_TOKEN is required")
    return token

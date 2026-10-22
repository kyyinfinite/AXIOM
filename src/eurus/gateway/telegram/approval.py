from __future__ import annotations

import asyncio
import contextvars
import json
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from eurus.models import ToolCall
from eurus.tools.base import Tool


@dataclass(frozen=True, slots=True)
class ApprovalContext:
    task_id: str
    owner_id: str
    chat_id: int


@dataclass(slots=True)
class PendingApproval:
    token: str
    task_id: str
    owner_id: str
    chat_id: int
    tool_name: str
    arguments_json: str
    future: asyncio.Future[bool]


Notifier = Callable[[int, str], Awaitable[None]]
ApprovalStateCallback = Callable[[str, bool], Awaitable[None]]

current_approval_context: contextvars.ContextVar[ApprovalContext | None] = (
    contextvars.ContextVar("eurus_approval_context", default=None)
)


class TelegramApprovalBroker:
    def __init__(
        self,
        *,
        timeout: float = 120.0,
        state_callback: ApprovalStateCallback | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("approval timeout must be positive")
        self.timeout = timeout
        self.state_callback = state_callback
        self.notifier: Notifier | None = None
        self.pending: dict[str, PendingApproval] = {}
        self._lock = asyncio.Lock()

    def bind_notifier(self, notifier: Notifier) -> None:
        self.notifier = notifier

    async def approve(self, call: ToolCall, tool: Tool) -> bool:
        context = current_approval_context.get()
        notifier = self.notifier
        state_callback = self.state_callback
        if context is None or notifier is None:
            return False
        token = secrets.token_urlsafe(9)
        loop = asyncio.get_running_loop()
        arguments = json.dumps(
            call.arguments,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        pending = PendingApproval(
            token=token,
            task_id=context.task_id,
            owner_id=context.owner_id,
            chat_id=context.chat_id,
            tool_name=tool.name,
            arguments_json=arguments[:1800],
            future=loop.create_future(),
        )
        async with self._lock:
            self.pending[token] = pending
        prompt = (
            "Owner approval required\n"
            f"Task: {context.task_id[:8]}\n"
            f"Tool: {tool.name}\n"
            f"Arguments: {pending.arguments_json}\n"
            f"Approve: /approve {token}\n"
            f"Deny: /deny {token}\n"
            f"Expires in {int(self.timeout)} seconds."
        )
        try:
            if state_callback is not None:
                await state_callback(context.task_id, True)
            await notifier(context.chat_id, prompt)
            return await asyncio.wait_for(pending.future, timeout=self.timeout)
        except TimeoutError:
            return False
        except asyncio.CancelledError:
            if not pending.future.done():
                pending.future.cancel()
            raise
        finally:
            async with self._lock:
                self.pending.pop(token, None)
            if state_callback is not None:
                await state_callback(context.task_id, False)

    async def resolve(
        self,
        token: str,
        *,
        owner_id: str,
        chat_id: int,
        allowed: bool,
    ) -> bool:
        async with self._lock:
            pending = self.pending.get(token)
            if pending is None:
                return False
            if pending.owner_id != owner_id or pending.chat_id != chat_id:
                return False
            if pending.future.done():
                return False
            pending.future.set_result(allowed)
            return True

    async def cancel_task(self, task_id: str) -> None:
        async with self._lock:
            matches = [item for item in self.pending.values() if item.task_id == task_id]
            for item in matches:
                if not item.future.done():
                    item.future.set_result(False)

    async def close(self) -> None:
        async with self._lock:
            pending = list(self.pending.values())
            self.pending.clear()
            for item in pending:
                if not item.future.done():
                    item.future.set_result(False)

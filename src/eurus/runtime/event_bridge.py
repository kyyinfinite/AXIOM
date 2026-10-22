from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

from eurus.agent.events import AgentEvent


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    session_id: str
    sequence: int
    event: AgentEvent


class AgentEventBridge:
    def __init__(self, *, default_capacity: int = 128) -> None:
        if default_capacity < 1:
            raise ValueError("event bridge capacity must be positive")
        self.default_capacity = default_capacity
        self._sequence = 0
        self._subscribers: dict[str, dict[str, asyncio.Queue[RuntimeEvent]]] = {}
        self.closed = False

    def subscribe(
        self, session_id: str, *, capacity: int | None = None
    ) -> tuple[str, asyncio.Queue[RuntimeEvent]]:
        if self.closed:
            raise RuntimeError("event bridge is closed")
        if not session_id.strip():
            raise ValueError("session id must not be empty")
        queue_capacity = self.default_capacity if capacity is None else capacity
        if queue_capacity < 1:
            raise ValueError("subscriber capacity must be positive")
        subscriber_id = uuid.uuid4().hex
        queue: asyncio.Queue[RuntimeEvent] = asyncio.Queue(maxsize=queue_capacity)
        self._subscribers.setdefault(session_id, {})[subscriber_id] = queue
        return subscriber_id, queue

    def unsubscribe(self, session_id: str, subscriber_id: str) -> bool:
        subscribers = self._subscribers.get(session_id)
        if subscribers is None or subscriber_id not in subscribers:
            return False
        del subscribers[subscriber_id]
        if not subscribers:
            del self._subscribers[session_id]
        return True

    async def publish(self, session_id: str, event: AgentEvent) -> None:
        if self.closed:
            return
        self._sequence += 1
        envelope = RuntimeEvent(session_id, self._sequence, event)
        for queue in tuple(self._subscribers.get(session_id, {}).values()):
            if queue.full():
                queue.get_nowait()
                queue.task_done()
            queue.put_nowait(envelope)

    def close(self) -> None:
        self.closed = True
        self._subscribers.clear()

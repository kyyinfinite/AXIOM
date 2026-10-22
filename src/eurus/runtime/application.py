from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

from eurus.agent.events import AgentEvent
from eurus.agent.loop import Agent
from eurus.runtime.event_bridge import AgentEventBridge
from eurus.sessions import SessionStore

AgentFactory = Callable[[str], Agent]
CloseCallback = Callable[[], Awaitable[None]]


class AgentRuntime:
    def __init__(
        self,
        agent_factory: AgentFactory,
        session_store: SessionStore,
        *,
        close_callback: CloseCallback | None = None,
        event_bridge: AgentEventBridge | None = None,
    ) -> None:
        self.agent_factory = agent_factory
        self.session_store = session_store
        self.close_callback = close_callback
        self.event_bridge = event_bridge or AgentEventBridge()
        self.agents: dict[str, Agent] = {}
        self.locks: dict[str, asyncio.Lock] = {}
        self.closed = False
        self._lifecycle_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._active_runs = 0
        self._idle = asyncio.Event()
        self._idle.set()

    def _agent(self, session_id: str) -> Agent:
        agent = self.agents.get(session_id)
        if agent is not None:
            return agent
        agent = self.agent_factory(session_id)
        agent.history.extend(self.session_store.load_messages(session_id))
        agent.session_id = session_id
        agent.context_store = self.session_store
        self.agents[session_id] = agent
        return agent

    async def run(self, session_id: str, text: str) -> AsyncIterator[AgentEvent]:
        if not session_id.strip():
            raise ValueError("session id must not be empty")
        if not text.strip():
            raise ValueError("message text must not be empty")
        async with self._lifecycle_lock:
            if self.closed:
                raise RuntimeError("agent runtime is closed")
            self._active_runs += 1
            self._idle.clear()
        try:
            lock = self.locks.setdefault(session_id, asyncio.Lock())
            async with lock:
                agent = self._agent(session_id)
                try:
                    async for event in agent.run(text):
                        await self.event_bridge.publish(session_id, event)
                        yield event
                finally:
                    self.session_store.save_messages(
                        session_id,
                        agent.history,
                        project_scope=str(agent.context.workspace.resolve()),
                    )
        finally:
            async with self._lifecycle_lock:
                self._active_runs -= 1
                if self._active_runs == 0:
                    self._idle.set()

    async def close(self) -> None:
        async with self._close_lock:
            if self.closed:
                return
            async with self._lifecycle_lock:
                self.closed = True
            await self._idle.wait()
            try:
                if self.close_callback is not None:
                    await self.close_callback()
            finally:
                self.agents.clear()
                self.locks.clear()
                self.event_bridge.close()

    async def __aenter__(self) -> AgentRuntime:
        if self.closed:
            raise RuntimeError("agent runtime is closed")
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

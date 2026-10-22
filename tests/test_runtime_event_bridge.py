from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from eurus.agent.events import AgentEvent
from eurus.models import Message, Usage
from eurus.runtime.application import AgentRuntime
from eurus.runtime.event_bridge import AgentEventBridge
from eurus.sessions import SessionStore


class EventAgent:
    def __init__(self, session_id: str, workspace: Path) -> None:
        self.session_id = session_id
        self.context_store = None
        self.context = SimpleNamespace(workspace=workspace)
        self.history = [Message.system("system")]

    async def run(self, text: str):
        self.history.extend((Message.user(text), Message.assistant(text)))
        yield AgentEvent.text_delta(text)
        yield AgentEvent.finished("stop", Usage())


@pytest.mark.asyncio
async def test_runtime_publishes_events_for_matching_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.sqlite3")
    bridge = AgentEventBridge(default_capacity=4)
    subscriber_id, queue = bridge.subscribe("session-a")
    runtime = AgentRuntime(
        lambda session_id: EventAgent(session_id, tmp_path),
        store,
        event_bridge=bridge,
    )

    events = [event async for event in runtime.run("session-a", "hello")]

    assert len(events) == 2
    first = queue.get_nowait()
    second = queue.get_nowait()
    assert first.session_id == "session-a"
    assert first.sequence < second.sequence
    assert first.event.kind == "text"
    assert bridge.unsubscribe("session-a", subscriber_id) is True
    await runtime.close()
    assert bridge.closed is True


@pytest.mark.asyncio
async def test_event_bridge_bounds_slow_subscribers() -> None:
    bridge = AgentEventBridge(default_capacity=1)
    _, queue = bridge.subscribe("session-a")

    await bridge.publish("session-a", AgentEvent.text_delta("first"))
    await bridge.publish("session-a", AgentEvent.text_delta("second"))

    event = queue.get_nowait()
    assert event.event.text == "second"
    assert queue.empty() is True


def test_event_bridge_rejects_invalid_capacity_and_closed_subscriptions() -> None:
    with pytest.raises(ValueError):
        AgentEventBridge(default_capacity=0)
    bridge = AgentEventBridge()
    bridge.close()
    with pytest.raises(RuntimeError):
        bridge.subscribe("session-a")


@pytest.mark.asyncio
async def test_runtime_close_waits_for_active_run(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    closed: list[bool] = []

    class BlockingAgent(EventAgent):
        async def run(self, text: str):
            started.set()
            await release.wait()
            async for event in super().run(text):
                yield event

    runtime = AgentRuntime(
        lambda session_id: BlockingAgent(session_id, tmp_path),
        SessionStore(tmp_path / "sessions.sqlite3"),
        close_callback=lambda: _mark_closed(closed),
    )
    run_task = asyncio.create_task(_collect_events(runtime))
    await started.wait()
    close_task = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)
    assert not close_task.done()
    release.set()
    await run_task
    await close_task
    assert closed == [True]
    assert runtime.closed is True


async def _collect_events(runtime: AgentRuntime) -> list[AgentEvent]:
    return [event async for event in runtime.run("session-a", "hello")]


async def _mark_closed(closed: list[bool]) -> None:
    closed.append(True)

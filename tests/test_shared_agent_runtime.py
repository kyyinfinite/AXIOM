from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from eurus.agent.events import AgentEvent
from eurus.models import Message
from eurus.runtime.application import AgentRuntime
from eurus.sessions import SessionStore


class FakeAgent:
    def __init__(self, session_id: str, workspace: Path) -> None:
        self.session_id = session_id
        self.context_store = None
        self.context = SimpleNamespace(workspace=workspace)
        self.history = [Message.system("system")]
        self.calls: list[str] = []

    async def run(self, text: str):
        self.calls.append(text)
        self.history.append(Message.user(text))
        self.history.append(Message.assistant(f"reply:{text}"))
        yield AgentEvent.text_delta(f"reply:{text}")


@pytest.mark.asyncio
async def test_runtime_reuses_agent_and_persists_history(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.sqlite3")
    created: list[FakeAgent] = []

    def factory(session_id: str) -> FakeAgent:
        agent = FakeAgent(session_id, tmp_path)
        created.append(agent)
        return agent

    runtime = AgentRuntime(factory, store)
    first = [event async for event in runtime.run("session-a", "first")]
    second = [event async for event in runtime.run("session-a", "second")]

    assert [event.text for event in first + second] == ["reply:first", "reply:second"]
    assert len(created) == 1
    assert created[0].calls == ["first", "second"]
    saved = store.load_messages("session-a")
    assert [message.content for message in saved] == [
        "first",
        "reply:first",
        "second",
        "reply:second",
    ]
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_serializes_same_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.sqlite3")
    runtime = AgentRuntime(lambda session_id: FakeAgent(session_id, tmp_path), store)
    active = 0
    maximum = 0
    agent = runtime._agent("session-a")

    async def run(text: str):
        nonlocal active, maximum
        lock = runtime.locks.setdefault("session-a", asyncio.Lock())
        async with lock:
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.01)
            active -= 1

    await asyncio.gather(run("one"), run("two"))
    assert maximum == 1
    assert agent.session_id == "session-a"
    await runtime.close()


@pytest.mark.asyncio
async def test_runtime_rejects_invalid_turns_and_closed_runtime(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.sqlite3")
    runtime = AgentRuntime(lambda session_id: FakeAgent(session_id, tmp_path), store)

    with pytest.raises(ValueError):
        _ = [event async for event in runtime.run("", "prompt")]
    with pytest.raises(ValueError):
        _ = [event async for event in runtime.run("session-a", " ")]

    await runtime.close()
    with pytest.raises(RuntimeError):
        _ = [event async for event in runtime.run("session-a", "prompt")]

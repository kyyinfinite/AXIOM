from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from eurus.agent.events import AgentEvent
from eurus.models import ToolCall, ToolResult
from eurus.trajectory import TrajectoryRecorder, TrajectoryStore, redact


def test_trajectory_persists_queries_and_redacts_secrets(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path / "trajectory.db", retention_days=30)
    event_id = store.record(
        "task-1",
        "tool.result",
        session_id="session-1",
        step_id="step-2",
        outcome="success",
        payload={"api_key": "should-not-persist", "result": "nvapi-abcdefgh1234567890"},
    )
    rows = store.query(task_id="task-1")
    assert rows[0]["id"] == event_id
    assert rows[0]["payload"]["api_key"] == "[REDACTED]"
    assert "nvapi-abcdefgh1234567890" not in store.export_jsonl()


def test_trajectory_validates_queries_and_prunes(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path / "trajectory.db", retention_days=1)
    store.record("task-1", "task.created")
    with pytest.raises(ValueError):
        store.query(limit=0)
    with pytest.raises(ValueError):
        store.query(since=datetime.now())
    removed = store.prune(now=datetime.now(UTC) + timedelta(days=3))
    assert removed == 1


def test_redaction_handles_nested_payloads() -> None:
    assert redact({"nested": {"password": "value"}}) == {"nested": {"password": "[REDACTED]"}}


def test_recorder_persists_agent_tool_events(tmp_path: Path) -> None:
    store = TrajectoryStore(tmp_path / "trajectory.db")
    recorder = TrajectoryRecorder(store, "task-2", "session-2")
    recorder.record_agent_event(
        AgentEvent.tool_started(ToolCall("call-1", "read_file", {"path": "a.txt"}))
    )
    recorder.record_agent_event(
        AgentEvent.tool_finished(ToolResult("call-1", "read_file", "contents", False))
    )
    rows = store.query(task_id="task-2")
    assert len(rows) == 2
    assert rows[0]["session_id"] == "session-2"
    assert rows[0]["event_type"] == "agent.tool_result"
    assert rows[0]["payload"]["result"] == "contents"


def test_agent_recorder_callback_records_yielded_events(tmp_path: Path) -> None:
    from eurus.agent.loop import Agent, IterationBudget
    from eurus.providers.testing import openai_message, scripted_router
    from eurus.tools.base import ToolContext, ToolRegistry

    router, _ = scripted_router(json_responses=[openai_message("finished")])
    store = TrajectoryStore(tmp_path / "agent.db")
    recorder = TrajectoryRecorder(store, "agent-task", "agent-session")
    agent = Agent(
        router,
        ToolRegistry(),
        ToolContext(tmp_path),
        system_prompt="test",
        stream=False,
        iteration_budget=IterationBudget(2),
        event_recorder=recorder.record_agent_event,
    )

    async def execute():
        return [event async for event in agent.run("hello")]

    import asyncio

    events = asyncio.run(execute())
    assert events[-1].kind == "done"
    assert store.query(task_id="agent-task")


def test_trajectory_records_runtime_events_with_redaction(tmp_path: Path) -> None:
    from types import SimpleNamespace

    store = TrajectoryStore(tmp_path / "runtime.db")
    recorder = TrajectoryRecorder(store, "runtime-task", "runtime-session")
    recorder.record_runtime_event(
        SimpleNamespace(
            kind="run.completed",
            run_id="run-1",
            status="completed",
            step_id="step-3",
            outcome="success",
            payload={"api_key": "secret-value", "tokens": 12},
        )
    )
    row = store.query(task_id="runtime-task")[0]
    assert row["event_type"] == "runtime.run.completed"
    assert row["payload"]["api_key"] == "[REDACTED]"
    assert row["payload"]["run_id"] == "run-1"

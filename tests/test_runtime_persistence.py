import sqlite3

import pytest

from eurus.runtime.engine import AgentEngine
from eurus.runtime.models import Run, RunStatus, RuntimeEvent
from eurus.runtime.recovery import RecoveryAction, classify_recovery
from eurus.runtime.state_machine import transition_run
from eurus.storage import ConcurrentRunUpdate, Database, EventStore, RunStore


def test_migration_is_additive_and_idempotent(tmp_path):
    path = tmp_path / "sessions.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT)")
        connection.execute("INSERT INTO sessions VALUES ('s1', 'existing')")
    database = Database(path)
    assert Database(path)
    with database.connection() as connection:
        row = connection.execute("SELECT title FROM sessions WHERE id = 's1'").fetchone()
        assert row[0] == "existing"
        versions = connection.execute("SELECT version FROM eurus_schema_migrations").fetchall()
        assert [row[0] for row in versions] == [1, 2, 3]
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert {"eurus_runs", "eurus_run_checkpoints", "eurus_runtime_events"} <= tables


def test_run_store_roundtrip_and_optimistic_update(tmp_path):
    store = RunStore(tmp_path / "runtime.sqlite3")
    run = Run(owner_id="test", objective="persist me")
    store.create(run)
    loaded = store.get(run.run_id)
    assert loaded is not None
    assert loaded.objective == "persist me"
    transition_run(loaded, RunStatus.RUNNING)
    store.save(loaded, expected_status_version=0)
    stale = store.get(run.run_id)
    assert stale is not None
    transition_run(loaded, RunStatus.FAILED, stop_reason="test")
    store.save(loaded, expected_status_version=1)
    with pytest.raises(ConcurrentRunUpdate):
        store.save(stale, expected_status_version=0)
    assert store.get(run.run_id).status == RunStatus.FAILED


def test_checkpoint_and_event_store_survive_store_recreation(tmp_path):
    path = tmp_path / "runtime.sqlite3"
    run_store = RunStore(path)
    event_store = EventStore(path)
    run = Run(owner_id="test", objective="checkpoint me")
    run_store.create(run)
    version = run_store.save_checkpoint(
        run.run_id,
        {"progress": {"step": 2}, "in_flight_side_effect": False},
    )
    loaded_run = run_store.get(run.run_id)
    assert loaded_run is not None
    transition_run(loaded_run, RunStatus.RUNNING)
    run_store.save(loaded_run, expected_status_version=0)
    event_store.append(
        RuntimeEvent(
            run_id=run.run_id,
            event_type="run.created",
            sequence_no=1,
            payload={"ok": True},
        )
    )
    reloaded_run_store = RunStore(path)
    reloaded_event_store = EventStore(path)
    checkpoint = reloaded_run_store.latest_checkpoint(run.run_id)
    events = reloaded_event_store.list_for_run(run.run_id)
    assert version == 1
    assert checkpoint is not None and checkpoint["state"]["progress"]["step"] == 2
    stored_after_checkpoint = reloaded_run_store.get(run.run_id)
    assert stored_after_checkpoint is not None
    assert stored_after_checkpoint.checkpoint_version == 1
    assert events[0].event_type == "run.created"
    assert events[0].payload == {"ok": True}


def test_checkpoint_rejects_sensitive_keys_and_stale_versions(tmp_path):
    store = RunStore(tmp_path / "runtime.sqlite3")
    run = Run(owner_id="test", objective="safe checkpoint")
    store.create(run)
    with pytest.raises(ValueError, match="sensitive field"):
        store.save_checkpoint(run.run_id, {"api_key": "not stored"})
    store.save_checkpoint(run.run_id, {"progress": 1})
    with pytest.raises(ConcurrentRunUpdate):
        store.save_checkpoint(run.run_id, {"progress": 2}, expected_version=0)


def test_recovery_classification_never_retries_unknown_side_effect():
    run = Run(owner_id="test", objective="recover me", status=RunStatus.RUNNING)
    assert classify_recovery(run, None) == RecoveryAction.MANUAL_REVIEW
    checkpoint = {"state": {"in_flight_side_effect": True}}
    assert classify_recovery(run, checkpoint) == RecoveryAction.RECONCILE_UNKNOWN_OUTCOME
    checkpoint = {"state": {"in_flight_side_effect": False, "progress": 3}}
    assert classify_recovery(run, checkpoint) == RecoveryAction.RESUME_CHECKPOINT
    queued = Run(owner_id="test", objective="fresh")
    assert classify_recovery(queued, None) == RecoveryAction.START_FRESH
    transition_run(queued, RunStatus.RUNNING)
    transition_run(queued, RunStatus.FAILED)
    assert classify_recovery(queued, None) == RecoveryAction.TERMINAL


@pytest.mark.asyncio
async def test_engine_persists_run_lifecycle_and_events(tmp_path):
    from eurus.agent.events import AgentEvent

    class FakeAgent:
        def __init__(self):
            self.max_parallel_tools = 4
            self.iteration_budget = type("Budget", (), {"limit": 50})()
            self.runtime_budget_tracker = None

        def interrupt(self):
            return None

        async def run(self, objective):
            yield AgentEvent.text_delta("durable result")
            yield AgentEvent.finished("completed", None)

    path = tmp_path / "runtime.sqlite3"
    run_store = RunStore(path)
    event_store = EventStore(path)
    result = await AgentEngine(FakeAgent(), event_store=event_store).run("durable objective")
    stored_run = run_store.get(result.run.run_id)
    stored_events = event_store.list_for_run(result.run.run_id)
    assert stored_run is not None and stored_run.status == RunStatus.COMPLETED
    assert [event.event_type for event in stored_events] == ["run.started", "run.completed"]
    assert [event.sequence_no for event in stored_events] == [1, 2]


@pytest.mark.asyncio
async def test_engine_continues_event_sequence_from_durable_event_store(tmp_path):
    from eurus.agent.events import AgentEvent

    class FakeAgent:
        max_parallel_tools = 4

        def __init__(self):
            self.iteration_budget = type("Budget", (), {"limit": 50})()
            self.runtime_budget_tracker = None

        def interrupt(self):
            return None

        async def run(self, objective):
            yield AgentEvent.finished("completed", None)

    path = tmp_path / "runtime.sqlite3"
    run_store = RunStore(path)
    event_store = EventStore(path)
    run = Run(owner_id="test", objective="sequence")
    run_store.create(run)
    event_store.append(RuntimeEvent(run_id=run.run_id, event_type="run.created", sequence_no=1))
    await AgentEngine(FakeAgent(), run_store=run_store, event_store=event_store).run(
        run.objective, run=run
    )
    events = event_store.list_for_run(run.run_id)
    assert [event.sequence_no for event in events] == [1, 2, 3]
    assert [event.event_type for event in events] == [
        "run.created",
        "run.started",
        "run.completed",
    ]


def test_database_requires_durable_file_backing():
    with pytest.raises(ValueError, match="file-backed SQLite"):
        Database(":memory:")


@pytest.mark.asyncio
async def test_persisted_operator_cancellation_interrupts_active_engine(tmp_path):
    import asyncio

    from eurus.agent.events import AgentEvent

    class InterruptibleAgent:
        def __init__(self):
            self.max_parallel_tools = 4
            self.iteration_budget = type("Budget", (), {"limit": 50})()
            self.runtime_budget_tracker = None
            self.interrupted = asyncio.Event()

        def interrupt(self):
            self.interrupted.set()

        async def run(self, objective):
            await self.interrupted.wait()
            yield AgentEvent.finished("interrupted", None)

    path = tmp_path / "runtime.sqlite3"
    runs = RunStore(path)
    events = EventStore(path)
    agent = InterruptibleAgent()
    engine = AgentEngine(agent, run_store=runs, event_store=events)
    task = asyncio.create_task(engine.run("cancel me"))
    for _ in range(100):
        await asyncio.sleep(0.01)
        active = runs.list_runs(limit=1)
        if active and active[0].status == RunStatus.RUNNING:
            break
    assert runs.request_cancel(active[0].run_id)
    result = await asyncio.wait_for(task, timeout=2)
    assert result.run.status == RunStatus.CANCELLED
    assert runs.get(result.run.run_id).status == RunStatus.CANCELLED
    assert any(
        event.event_type == "run.cancelled" for event in events.list_for_run(result.run.run_id)
    )

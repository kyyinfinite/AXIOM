from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from eurus.automation.worker import AutomationWorkerPool
from eurus.scheduler import ScheduleStore
from eurus.storage.jobs import JobStore


def test_job_claim_is_exclusive_and_lease_fences_stale_worker(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    now = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
    job = store.enqueue("owner", "task", available_at=now)
    claimed = store.claim("worker-a", now=now, lease_seconds=10)
    assert claimed is not None
    assert claimed.job_id == job.job_id
    assert store.claim("worker-b", now=now, lease_seconds=10) is None
    assert not store.finish(
        job.job_id,
        "worker-b",
        claimed.lease_version,
        status="completed",
        now=now + timedelta(seconds=1),
    )
    assert store.finish(
        job.job_id,
        "worker-a",
        claimed.lease_version,
        status="completed",
        result="done",
        now=now + timedelta(seconds=1),
    )
    assert store.get(job.job_id).status == "completed"


def test_expired_lease_requeues_only_before_side_effect_boundary(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    now = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
    safe = store.enqueue("owner", "safe", available_at=now, max_attempts=2)
    uncertain = store.enqueue("owner", "uncertain", available_at=now)
    safe_claim = store.claim("worker-a", now=now, lease_seconds=5)
    uncertain_claim = store.claim("worker-b", now=now, lease_seconds=5)
    assert safe_claim is not None and uncertain_claim is not None
    assert store.mark_side_effect_started(
        uncertain.job_id, "worker-b", uncertain_claim.lease_version
    )
    outcome = store.recover_expired(now=now + timedelta(seconds=6))
    assert outcome == {"requeued": 1, "unknown_outcome": 1, "failed": 0}
    assert store.get(safe.job_id).status == "queued"
    assert store.get(uncertain.job_id).status == "unknown_outcome"


def test_dispatch_due_persists_job_and_skips_unbounded_catchup(tmp_path: Path) -> None:
    path = tmp_path / "automation.sqlite3"
    store = ScheduleStore(path)
    JobStore(path)
    now = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
    schedule = store.create(
        "owner",
        "scheduled task",
        kind="interval",
        interval_seconds=60,
        now=now,
    )
    due = now + timedelta(hours=2)
    job_ids = store.dispatch_due(now=due, owner_id="owner")
    assert len(job_ids) == 1
    assert store.dispatch_due(now=due, owner_id="owner") == []
    job = JobStore(path).get(job_ids[0])
    assert job is not None
    assert job.status == "queued"
    assert job.schedule_id == schedule.schedule_id
    assert store.history(schedule.schedule_id)[0].status == "queued"
    restored = store.get(schedule.schedule_id)
    assert restored is not None
    assert datetime.fromisoformat(restored.next_run_at) == due + timedelta(seconds=60)


def test_worker_pool_processes_jobs_concurrently(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = JobStore(tmp_path / "workers.sqlite3")
        for index in range(3):
            store.enqueue("owner", f"task-{index}")
        active = 0
        maximum_active = 0

        async def handler(job) -> str:
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.03)
            active -= 1
            return job.objective

        pool = AutomationWorkerPool(store, handler, workers=3, lease_seconds=5)
        await asyncio.gather(
            pool.run_once("worker-a"),
            pool.run_once("worker-b"),
            pool.run_once("worker-c"),
        )
        assert maximum_active == 3
        assert all(job.status == "completed" for job in store.list_jobs(owner_id="owner", limit=10))

    asyncio.run(scenario())


def test_job_cancel_is_owner_scoped(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "cancel.sqlite3")
    job = store.enqueue("alice", "queued")
    assert not store.request_cancel(job.job_id, owner_id="bob")
    assert store.request_cancel(job.job_id, owner_id="alice")
    assert store.get(job.job_id).status == "cancelled"

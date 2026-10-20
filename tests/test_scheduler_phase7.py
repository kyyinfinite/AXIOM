from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from eurus.scheduler import ScheduleStore, TaskProfile


def test_interval_schedule_persists_and_uses_bounded_catchup(tmp_path: Path) -> None:
    path = tmp_path / "schedule.sqlite3"
    store = ScheduleStore(path)
    now = datetime(2026, 10, 19, 10, 0, tzinfo=UTC)
    schedule = store.create(
        "user-1",
        "check project",
        kind="interval",
        interval_seconds=60,
        timezone="Asia/Jakarta",
        now=now,
    )
    restored = ScheduleStore(path).get(schedule.schedule_id, owner_id="user-1")
    assert restored is not None
    assert restored.next_run_at == (now + timedelta(seconds=60)).isoformat()
    runs = store.claim_due(now=now + timedelta(hours=3))
    assert len(runs) == 1
    updated = store.get(schedule.schedule_id)
    assert updated is not None
    assert updated.next_run_at == (now + timedelta(hours=3, seconds=60)).isoformat()
    assert not store.claim_due(now=now + timedelta(hours=3))


def test_one_time_schedule_runs_once_and_history_persists(tmp_path: Path) -> None:
    store = ScheduleStore(tmp_path / "schedule.sqlite3")
    now = datetime(2026, 10, 19, 10, 0, tzinfo=UTC)
    schedule = store.create(
        "u",
        "prepare report",
        kind="once",
        run_at="2026-10-19T18:00:00+07:00",
        timezone="Asia/Jakarta",
        now=now,
    )
    run = store.claim_due(now=now + timedelta(hours=1))[0]
    assert run.schedule_id == schedule.schedule_id
    assert store.get(schedule.schedule_id).enabled is False
    assert store.finish_run(run.run_id, status="completed", result="verified")
    assert not store.finish_run(run.run_id, status="failed")
    assert ScheduleStore(store.path).history(schedule.schedule_id)[0].result == "verified"


def test_invalid_timezone_interval_and_profile_are_rejected(tmp_path: Path) -> None:
    store = ScheduleStore(tmp_path / "schedule.sqlite3")
    with pytest.raises(ValueError, match="timezone"):
        store.create("u", "x", kind="interval", interval_seconds=60, timezone="Mars/Base")
    with pytest.raises(ValueError, match="interval"):
        store.create("u", "x", kind="interval", interval_seconds=10)
    with pytest.raises(ValueError, match="timezone"):
        TaskProfile("bad", timezone="Mars/Base").validate()


def test_owner_scoped_schedule_controls_and_profiles(tmp_path: Path) -> None:
    store = ScheduleStore(tmp_path / "schedule.sqlite3")
    profile = TaskProfile("daily", timezone="Asia/Jakarta", max_steps=12)
    store.save_profile("alice", profile)
    assert store.get_profile("alice", "daily") == profile
    assert store.get_profile("bob", "daily") is None
    schedule = store.create("alice", "review", kind="interval", interval_seconds=3600)
    assert not store.set_enabled(schedule.schedule_id, "bob", False)
    assert store.set_enabled(schedule.schedule_id, "alice", False)
    assert not store.get(schedule.schedule_id).enabled


def test_due_claim_is_atomic_and_does_not_duplicate(tmp_path: Path) -> None:
    store = ScheduleStore(tmp_path / "schedule.sqlite3")
    now = datetime(2026, 10, 19, 10, 0, tzinfo=UTC)
    store.create("u", "once", kind="once", run_at="2026-10-19T10:01:00+00:00", now=now)
    due = now + timedelta(minutes=1)
    assert len(store.claim_due(now=due)) == 1
    assert store.claim_due(now=due) == []


def test_schedule_cli_create_list_and_profile(tmp_path: Path, monkeypatch) -> None:
    from click.testing import CliRunner

    from eurus.cli import main

    monkeypatch.setenv("EURUS_HOME", str(tmp_path / ".eurus"))
    runner = CliRunner()
    profile = runner.invoke(main, ["schedule", "profile-save", "focus", "--timezone", "UTC"])
    assert profile.exit_code == 0, profile.output
    added = runner.invoke(
        main, ["schedule", "add", "review", "--every", "3600", "--profile", "focus"]
    )
    assert added.exit_code == 0, added.output
    listed = runner.invoke(main, ["schedule", "list"])
    assert listed.exit_code == 0
    assert "review" in listed.output

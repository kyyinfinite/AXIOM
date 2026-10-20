from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

LATEST_SCHEMA_VERSION = 3

_MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        """CREATE TABLE eurus_runs (
            run_id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            objective TEXT NOT NULL,
            parent_run_id TEXT,
            session_id TEXT,
            status TEXT NOT NULL,
            stop_reason TEXT,
            budget_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            schema_version INTEGER NOT NULL,
            checkpoint_version INTEGER NOT NULL,
            result_ref TEXT,
            status_version INTEGER NOT NULL
        )""",
        "CREATE INDEX eurus_runs_owner_created ON eurus_runs(owner_id, created_at DESC)",
        "CREATE INDEX eurus_runs_status_updated ON eurus_runs(status, updated_at)",
        """CREATE TABLE eurus_run_checkpoints (
            run_id TEXT NOT NULL REFERENCES eurus_runs(run_id) ON DELETE CASCADE,
            checkpoint_version INTEGER NOT NULL,
            checkpoint_type TEXT NOT NULL,
            state_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, checkpoint_version)
        )""",
        """CREATE TABLE eurus_runtime_events (
            event_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES eurus_runs(run_id) ON DELETE CASCADE,
            sequence_no INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            payload_version INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            step_id TEXT,
            attempt_id TEXT,
            parent_run_id TEXT,
            UNIQUE (run_id, sequence_no)
        )""",
        "CREATE INDEX eurus_runtime_events_run_sequence "
        "ON eurus_runtime_events(run_id, sequence_no)",
        "CREATE INDEX eurus_runtime_events_type_created "
        "ON eurus_runtime_events(event_type, created_at)",
    ),
    2: (
        """CREATE TABLE eurus_jobs (
            job_id TEXT PRIMARY KEY,
            schedule_id TEXT,
            scheduled_at TEXT,
            owner_id TEXT NOT NULL,
            objective TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 1,
            available_at TEXT NOT NULL,
            worker_id TEXT,
            lease_expires_at TEXT,
            lease_version INTEGER NOT NULL DEFAULT 0,
            side_effect_started INTEGER NOT NULL DEFAULT 0,
            cancel_requested INTEGER NOT NULL DEFAULT 0,
            result TEXT NOT NULL DEFAULT '',
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(schedule_id, scheduled_at)
        )""",
        "CREATE INDEX eurus_jobs_claimable ON eurus_jobs(status, available_at, created_at)",
        "CREATE INDEX eurus_jobs_lease ON eurus_jobs(status, lease_expires_at)",
        "CREATE INDEX eurus_jobs_schedule ON eurus_jobs(schedule_id, created_at DESC)",
    ),
    3: (
        """CREATE TABLE eurus_run_controls (
            run_id TEXT PRIMARY KEY REFERENCES eurus_runs(run_id) ON DELETE CASCADE,
            cancel_requested INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )""",
    ),
}


def apply_migrations(connection: sqlite3.Connection) -> int:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS eurus_schema_migrations "
        "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    connection.commit()
    applied = {
        int(row[0]) for row in connection.execute("SELECT version FROM eurus_schema_migrations")
    }
    if max(applied, default=0) > LATEST_SCHEMA_VERSION:
        raise RuntimeError("database schema is newer than this Eurus version")
    for version in sorted(_MIGRATIONS):
        if version in applied:
            continue
        connection.execute("BEGIN IMMEDIATE")
        try:
            already_applied = connection.execute(
                "SELECT 1 FROM eurus_schema_migrations WHERE version = ?", (version,)
            ).fetchone()
            if already_applied is not None:
                connection.commit()
                continue
            for statement in _MIGRATIONS[version]:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO eurus_schema_migrations(version, applied_at) VALUES (?, ?)",
                (version, datetime.now(UTC).isoformat()),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
    return max(applied | set(_MIGRATIONS), default=0)

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from eurus.storage.database import Database, DatabasePath

_JOB_STATUSES = frozenset(
    {"queued", "running", "completed", "failed", "cancelled", "unknown_outcome"}
)


def _now(value: datetime | None = None) -> datetime:
    current = value or datetime.now(UTC)
    if current.tzinfo is None:
        raise ValueError("datetime must include a timezone")
    return current.astimezone(UTC)


def _time(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class Job:
    job_id: str
    owner_id: str
    objective: str
    status: str
    schedule_id: str | None
    scheduled_at: str | None
    payload: dict[str, Any]
    attempts: int
    max_attempts: int
    available_at: str
    worker_id: str | None
    lease_expires_at: str | None
    lease_version: int
    side_effect_started: bool
    cancel_requested: bool
    result: str
    error: str | None
    created_at: str
    updated_at: str


def _job(row: sqlite3.Row) -> Job:
    return Job(
        job_id=str(row["job_id"]),
        owner_id=str(row["owner_id"]),
        objective=str(row["objective"]),
        status=str(row["status"]),
        schedule_id=row["schedule_id"],
        scheduled_at=row["scheduled_at"],
        payload=json.loads(row["payload_json"]),
        attempts=int(row["attempts"]),
        max_attempts=int(row["max_attempts"]),
        available_at=str(row["available_at"]),
        worker_id=row["worker_id"],
        lease_expires_at=row["lease_expires_at"],
        lease_version=int(row["lease_version"]),
        side_effect_started=bool(row["side_effect_started"]),
        cancel_requested=bool(row["cancel_requested"]),
        result=str(row["result"]),
        error=row["error"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


class JobStore:
    def __init__(self, database: Database | DatabasePath) -> None:
        self.database = database if isinstance(database, Database) else Database(database)

    def enqueue(
        self,
        owner_id: str,
        objective: str,
        *,
        payload: dict[str, Any] | None = None,
        schedule_id: str | None = None,
        scheduled_at: str | None = None,
        max_attempts: int = 1,
        available_at: datetime | None = None,
        job_id: str | None = None,
    ) -> Job:
        if not owner_id.strip() or not objective.strip():
            raise ValueError("owner_id and objective are required")
        if not 1 <= max_attempts <= 20:
            raise ValueError("max_attempts must be between 1 and 20")
        if (schedule_id is None) != (scheduled_at is None):
            raise ValueError("schedule_id and scheduled_at must be provided together")
        current = _now()
        identifier = job_id or uuid.uuid4().hex
        available = _time(_now(available_at)) if available_at is not None else _time(current)
        serialized = json.dumps(
            payload or {}, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        with self.database.connection() as connection:
            connection.execute(
                """INSERT INTO eurus_jobs(
                    job_id, schedule_id, scheduled_at, owner_id, objective, payload_json,
                    status, attempts, max_attempts, available_at, lease_version,
                    side_effect_started, cancel_requested, result, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'queued', 0, ?, ?, 0, 0, 0, '', ?, ?)""",
                (
                    identifier,
                    schedule_id,
                    scheduled_at,
                    owner_id,
                    objective,
                    serialized,
                    max_attempts,
                    available,
                    _time(current),
                    _time(current),
                ),
            )
            row = connection.execute(
                "SELECT * FROM eurus_jobs WHERE job_id=?", (identifier,)
            ).fetchone()
        if row is None:
            raise RuntimeError("enqueued job could not be read back")
        return _job(row)

    def get(self, job_id: str) -> Job | None:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT * FROM eurus_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        return _job(row) if row is not None else None

    def list_jobs(
        self, *, owner_id: str | None = None, statuses: tuple[str, ...] = (), limit: int = 100
    ) -> list[Job]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        clauses: list[str] = []
        values: list[Any] = []
        if owner_id is not None:
            clauses.append("owner_id=?")
            values.append(owner_id)
        if statuses:
            invalid = set(statuses) - _JOB_STATUSES
            if invalid:
                raise ValueError("invalid job status")
            clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
            values.extend(statuses)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.database.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM eurus_jobs{where} ORDER BY created_at DESC LIMIT ?",
                (*values, limit),
            ).fetchall()
        return [_job(row) for row in rows]

    def claim(
        self,
        worker_id: str,
        *,
        now: datetime | None = None,
        lease_seconds: int = 60,
    ) -> Job | None:
        if not worker_id.strip():
            raise ValueError("worker_id is required")
        if not 5 <= lease_seconds <= 86400:
            raise ValueError("lease_seconds must be between 5 and 86400")
        current = _now(now)
        current_text = _time(current)
        expires = _time(current + timedelta(seconds=lease_seconds))
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT job_id FROM eurus_jobs WHERE status='queued' AND available_at<=? "
                "ORDER BY available_at, created_at LIMIT 1",
                (current_text,),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            cursor = connection.execute(
                "UPDATE eurus_jobs SET status='running', attempts=attempts+1, worker_id=?, "
                "lease_expires_at=?, lease_version=lease_version+1, updated_at=? "
                "WHERE job_id=? AND status='queued'",
                (worker_id, expires, current_text, row["job_id"]),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            claimed = connection.execute(
                "SELECT * FROM eurus_jobs WHERE job_id=?", (row["job_id"],)
            ).fetchone()
            connection.commit()
        if claimed is None:
            raise RuntimeError("claimed job could not be read back")
        return _job(claimed)

    def heartbeat(
        self,
        job_id: str,
        worker_id: str,
        lease_version: int,
        *,
        now: datetime | None = None,
        lease_seconds: int = 60,
    ) -> bool:
        current = _now(now)
        expires = _time(current + timedelta(seconds=lease_seconds))
        with self.database.connection() as connection:
            cursor = connection.execute(
                "UPDATE eurus_jobs SET lease_expires_at=?, updated_at=? WHERE job_id=? "
                "AND status='running' AND worker_id=? AND lease_version=? "
                "AND lease_expires_at>=? AND cancel_requested=0",
                (expires, _time(current), job_id, worker_id, lease_version, _time(current)),
            )
        return cursor.rowcount == 1

    def mark_side_effect_started(self, job_id: str, worker_id: str, lease_version: int) -> bool:
        with self.database.connection() as connection:
            cursor = connection.execute(
                "UPDATE eurus_jobs SET side_effect_started=1, updated_at=? WHERE job_id=? "
                "AND status='running' AND worker_id=? AND lease_version=?",
                (_time(_now()), job_id, worker_id, lease_version),
            )
        return cursor.rowcount == 1

    def finish(
        self,
        job_id: str,
        worker_id: str,
        lease_version: int,
        *,
        status: str,
        result: str = "",
        error: str | None = None,
        now: datetime | None = None,
    ) -> bool:
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError("invalid terminal job status")
        current = _time(_now(now))
        with self.database.connection() as connection:
            cursor = connection.execute(
                "UPDATE eurus_jobs SET status=CASE WHEN cancel_requested=1 "
                "THEN 'cancelled' ELSE ? END, "
                "result=?, error=?, worker_id=NULL, lease_expires_at=NULL, updated_at=? "
                "WHERE job_id=? AND status='running' "
                "AND worker_id=? AND lease_version=? AND lease_expires_at>=?",
                (status, result[:4000], error, current, job_id, worker_id, lease_version, current),
            )
        return cursor.rowcount == 1

    def cancel(self, job_id: str, *, owner_id: str) -> bool:
        with self.database.connection() as connection:
            cursor = connection.execute(
                "UPDATE eurus_jobs SET status='cancelled', worker_id=NULL, lease_expires_at=NULL, "
                "updated_at=? WHERE job_id=? AND owner_id=? AND status='queued'",
                (_time(_now()), job_id, owner_id),
            )
        return cursor.rowcount == 1

    def request_cancel(self, job_id: str, *, owner_id: str) -> bool:
        current = _time(_now())
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE eurus_jobs SET status='cancelled', updated_at=? WHERE job_id=? "
                "AND owner_id=? AND status='queued'",
                (current, job_id, owner_id),
            )
            if cursor.rowcount == 0:
                cursor = connection.execute(
                    "UPDATE eurus_jobs SET cancel_requested=1, updated_at=? WHERE job_id=? "
                    "AND owner_id=? AND status='running' AND cancel_requested=0",
                    (current, job_id, owner_id),
                )
            connection.commit()
        return cursor.rowcount == 1

    def recover_expired(self, *, now: datetime | None = None) -> dict[str, int]:
        current = _now(now)
        current_text = _time(current)
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            safe = connection.execute(
                "UPDATE eurus_jobs SET status='queued', worker_id=NULL, lease_expires_at=NULL, "
                "available_at=?, updated_at=? WHERE status='running' AND lease_expires_at<? "
                "AND side_effect_started=0 AND attempts<max_attempts",
                (current_text, current_text, current_text),
            ).rowcount
            uncertain = connection.execute(
                "UPDATE eurus_jobs SET status='unknown_outcome', worker_id=NULL, "
                "lease_expires_at=NULL, error='lease_expired_after_side_effect_boundary', "
                "updated_at=? WHERE status='running' AND lease_expires_at<? "
                "AND side_effect_started=1",
                (current_text, current_text),
            ).rowcount
            exhausted = connection.execute(
                "UPDATE eurus_jobs SET status='failed', worker_id=NULL, lease_expires_at=NULL, "
                "error='lease_expired_attempt_limit', updated_at=? WHERE status='running' "
                "AND lease_expires_at<? AND side_effect_started=0 AND attempts>=max_attempts",
                (current_text, current_text),
            ).rowcount
            connection.commit()
        return {"requeued": safe, "unknown_outcome": uncertain, "failed": exhausted}

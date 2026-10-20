from __future__ import annotations

import json
import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ScheduleKind = Literal["once", "interval"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("datetime must include a timezone")
    return value.astimezone(UTC).isoformat()


def _parse_time(value: str, zone: ZoneInfo) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("run time must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=zone)
    return parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class TaskProfile:
    name: str
    timezone: str = "UTC"
    max_runtime_seconds: int = 1800
    max_steps: int = 20
    notifications: bool = True
    allowed_tools: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("profile name is required")
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("unknown IANA timezone") from exc
        if not 1 <= self.max_runtime_seconds <= 86400:
            raise ValueError("max_runtime_seconds must be between 1 and 86400")
        if not 1 <= self.max_steps <= 1000:
            raise ValueError("max_steps must be between 1 and 1000")
        if any(not item or len(item) > 128 for item in self.allowed_tools):
            raise ValueError("invalid allowed tool name")


@dataclass(frozen=True, slots=True)
class Schedule:
    schedule_id: str
    owner_id: str
    objective: str
    kind: ScheduleKind
    timezone: str
    next_run_at: str
    interval_seconds: int | None
    enabled: bool
    profile: dict[str, Any]
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ScheduleRun:
    run_id: str
    schedule_id: str
    scheduled_at: str
    status: str
    started_at: str
    finished_at: str | None
    result: str


class ScheduleStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.touch(mode=0o600, exist_ok=True)
        with suppress(OSError):
            os.chmod(self.path, 0o600)
        with self._connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS eurus_schedules (
                    schedule_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
                    objective TEXT NOT NULL, kind TEXT NOT NULL, timezone TEXT NOT NULL,
                    next_run_at TEXT NOT NULL, interval_seconds INTEGER,
                    enabled INTEGER NOT NULL, profile_json TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS eurus_schedules_due
                    ON eurus_schedules(enabled, next_run_at);
                CREATE TABLE IF NOT EXISTS eurus_schedule_runs (
                    run_id TEXT PRIMARY KEY, schedule_id TEXT NOT NULL,
                    scheduled_at TEXT NOT NULL, status TEXT NOT NULL,
                    started_at TEXT NOT NULL, finished_at TEXT, result TEXT NOT NULL DEFAULT '',
                    UNIQUE(schedule_id, scheduled_at),
                    FOREIGN KEY(schedule_id) REFERENCES eurus_schedules(schedule_id)
                );
                CREATE INDEX IF NOT EXISTS eurus_schedule_runs_schedule
                    ON eurus_schedule_runs(schedule_id, started_at);
                CREATE TABLE IF NOT EXISTS eurus_task_profiles (
                    owner_id TEXT NOT NULL, name TEXT NOT NULL, profile_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL, PRIMARY KEY(owner_id, name)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=10000")
        db.execute("PRAGMA foreign_keys=ON")
        return db

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        db = self._connect()
        try:
            yield db
        finally:
            db.close()

    def save_profile(self, owner_id: str, profile: TaskProfile) -> None:
        profile.validate()
        payload = json.dumps(
            {
                "timezone": profile.timezone,
                "max_runtime_seconds": profile.max_runtime_seconds,
                "max_steps": profile.max_steps,
                "notifications": profile.notifications,
                "allowed_tools": list(profile.allowed_tools),
            }
        )
        with self._connection() as db:
            db.execute(
                "INSERT INTO eurus_task_profiles VALUES (?, ?, ?, ?) "
                "ON CONFLICT(owner_id, name) DO UPDATE SET "
                "profile_json=excluded.profile_json, updated_at=excluded.updated_at",
                (owner_id, profile.name, payload, _iso(_utcnow())),
            )

    def get_profile(self, owner_id: str, name: str) -> TaskProfile | None:
        with self._connection() as db:
            row = db.execute(
                "SELECT profile_json FROM eurus_task_profiles WHERE owner_id=? AND name=?",
                (owner_id, name),
            ).fetchone()
        if row is None:
            return None
        values = json.loads(row["profile_json"])
        values["allowed_tools"] = tuple(values.get("allowed_tools", ()))
        profile = TaskProfile(name=name, **values)
        profile.validate()
        return profile

    def create(
        self,
        owner_id: str,
        objective: str,
        *,
        kind: ScheduleKind,
        run_at: str | None = None,
        interval_seconds: int | None = None,
        timezone: str = "UTC",
        profile: TaskProfile | None = None,
        now: datetime | None = None,
    ) -> Schedule:
        if not owner_id.strip() or not objective.strip():
            raise ValueError("owner and objective are required")
        try:
            zone = ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("unknown IANA timezone") from exc
        current = now or _utcnow()
        if current.tzinfo is None:
            raise ValueError("now must include a timezone")
        if kind == "once":
            if run_at is None:
                raise ValueError("one-time schedule requires run_at")
            next_run = _parse_time(run_at, zone)
            if next_run <= current.astimezone(UTC):
                raise ValueError("run_at must be in the future")
            interval = None
        elif kind == "interval":
            if interval_seconds is None or not 60 <= interval_seconds <= 31_536_000:
                raise ValueError("interval must be between 60 seconds and one year")
            next_run = current.astimezone(UTC) + timedelta(seconds=interval_seconds)
            interval = interval_seconds
        else:
            raise ValueError("unsupported schedule kind")
        selected_profile = profile or TaskProfile(name="default", timezone=timezone)
        selected_profile.validate()
        now_text = _iso(current)
        schedule = Schedule(
            uuid.uuid4().hex,
            owner_id,
            objective,
            kind,
            timezone,
            _iso(next_run),
            interval,
            True,
            {
                "name": selected_profile.name,
                "timezone": selected_profile.timezone,
                "max_runtime_seconds": selected_profile.max_runtime_seconds,
                "max_steps": selected_profile.max_steps,
                "notifications": selected_profile.notifications,
                "allowed_tools": list(selected_profile.allowed_tools),
            },
            now_text,
            now_text,
        )
        with self._connection() as db:
            db.execute(
                "INSERT INTO eurus_schedules VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    schedule.schedule_id,
                    owner_id,
                    objective,
                    kind,
                    timezone,
                    schedule.next_run_at,
                    interval,
                    1,
                    json.dumps(schedule.profile),
                    now_text,
                    now_text,
                ),
            )
        return schedule

    def get(self, schedule_id: str, *, owner_id: str | None = None) -> Schedule | None:
        with self._connection() as db:
            if owner_id is None:
                row = db.execute(
                    "SELECT * FROM eurus_schedules WHERE schedule_id=?", (schedule_id,)
                ).fetchone()
            else:
                row = db.execute(
                    "SELECT * FROM eurus_schedules WHERE schedule_id=? AND owner_id=?",
                    (schedule_id, owner_id),
                ).fetchone()
        return self._schedule(row) if row else None

    def list_owner(self, owner_id: str, limit: int = 100) -> list[Schedule]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        with self._connection() as db:
            rows = db.execute(
                "SELECT * FROM eurus_schedules WHERE owner_id=? ORDER BY created_at DESC LIMIT ?",
                (owner_id, limit),
            ).fetchall()
        return [self._schedule(row) for row in rows]

    def set_enabled(self, schedule_id: str, owner_id: str, enabled: bool) -> bool:
        with self._connection() as db:
            cursor = db.execute(
                "UPDATE eurus_schedules SET enabled=?, updated_at=? "
                "WHERE schedule_id=? AND owner_id=?",
                (int(enabled), _iso(_utcnow()), schedule_id, owner_id),
            )
            return cursor.rowcount == 1

    def claim_due(
        self, *, now: datetime | None = None, limit: int = 20, owner_id: str | None = None
    ) -> list[ScheduleRun]:
        current = now or _utcnow()
        if current.tzinfo is None:
            raise ValueError("now must include a timezone")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        now_text = _iso(current)
        claimed: list[ScheduleRun] = []
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            query = "SELECT * FROM eurus_schedules WHERE enabled=1 AND next_run_at<=?"
            args: tuple[Any, ...] = (now_text,)
            if owner_id is not None:
                query += " AND owner_id=?"
                args += (owner_id,)
            query += " ORDER BY next_run_at LIMIT ?"
            rows = db.execute(query, args + (limit,)).fetchall()
            for row in rows:
                scheduled_at = str(row["next_run_at"])
                run_id = uuid.uuid4().hex
                started = now_text
                try:
                    db.execute(
                        "INSERT INTO eurus_schedule_runs "
                        "(run_id, schedule_id, scheduled_at, status, started_at) "
                        "VALUES (?, ?, ?, 'running', ?)",
                        (run_id, row["schedule_id"], scheduled_at, started),
                    )
                except sqlite3.IntegrityError:
                    continue
                next_run = None
                if row["kind"] == "interval":
                    interval = int(row["interval_seconds"])
                    scheduled = datetime.fromisoformat(scheduled_at)
                    next_run = _iso(scheduled + timedelta(seconds=interval))
                    if datetime.fromisoformat(next_run) <= current.astimezone(UTC):
                        next_run = _iso(current.astimezone(UTC) + timedelta(seconds=interval))
                db.execute(
                    "UPDATE eurus_schedules SET enabled=?, next_run_at=COALESCE(?, next_run_at), "
                    "updated_at=? WHERE schedule_id=?",
                    (int(row["kind"] != "once"), next_run, now_text, row["schedule_id"]),
                )
                claimed.append(
                    ScheduleRun(
                        run_id, str(row["schedule_id"]), scheduled_at, "running", started, None, ""
                    )
                )
            db.commit()
        return claimed

    def dispatch_due(
        self, *, now: datetime | None = None, limit: int = 20, owner_id: str | None = None
    ) -> list[str]:
        current = now or _utcnow()
        if current.tzinfo is None:
            raise ValueError("now must include a timezone")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        now_text = _iso(current)
        job_ids: list[str] = []
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            query = "SELECT * FROM eurus_schedules WHERE enabled=1 AND next_run_at<=?"
            values: tuple[Any, ...] = (now_text,)
            if owner_id is not None:
                query += " AND owner_id=?"
                values += (owner_id,)
            query += " ORDER BY next_run_at LIMIT ?"
            rows = db.execute(query, values + (limit,)).fetchall()
            for row in rows:
                scheduled_at = str(row["next_run_at"])
                job_id = uuid.uuid4().hex
                payload = json.dumps(
                    {"profile": json.loads(row["profile_json"]), "timezone": row["timezone"]},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                cursor = db.execute(
                    """INSERT OR IGNORE INTO eurus_jobs(
                        job_id, schedule_id, scheduled_at, owner_id, objective, payload_json,
                        status, attempts, max_attempts, available_at, lease_version,
                        side_effect_started, cancel_requested, result, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'queued', 0, 1, ?, 0, 0, 0, '', ?, ?)""",
                    (
                        job_id,
                        row["schedule_id"],
                        scheduled_at,
                        row["owner_id"],
                        row["objective"],
                        payload,
                        now_text,
                        now_text,
                        now_text,
                    ),
                )
                if cursor.rowcount != 1:
                    continue
                db.execute(
                    "INSERT INTO eurus_schedule_runs "
                    "(run_id, schedule_id, scheduled_at, status, started_at) "
                    "VALUES (?, ?, ?, 'queued', ?)",
                    (job_id, row["schedule_id"], scheduled_at, now_text),
                )
                next_run = None
                if row["kind"] == "interval":
                    interval = int(row["interval_seconds"])
                    scheduled = datetime.fromisoformat(scheduled_at)
                    next_occurrence = scheduled + timedelta(seconds=interval)
                    if next_occurrence <= current.astimezone(UTC):
                        next_occurrence = current.astimezone(UTC) + timedelta(seconds=interval)
                    next_run = _iso(next_occurrence)
                db.execute(
                    "UPDATE eurus_schedules SET enabled=?, next_run_at=COALESCE(?, next_run_at), "
                    "updated_at=? WHERE schedule_id=?",
                    (int(row["kind"] != "once"), next_run, now_text, row["schedule_id"]),
                )
                job_ids.append(job_id)
            db.commit()
        return job_ids

    def mark_run_started(self, run_id: str, *, now: datetime | None = None) -> bool:
        current = _iso(now or _utcnow())
        with self._connection() as db:
            cursor = db.execute(
                "UPDATE eurus_schedule_runs SET status='running', started_at=? "
                "WHERE run_id=? AND status='queued'",
                (current, run_id),
            )
            return cursor.rowcount == 1

    def finish_run(self, run_id: str, *, status: str, result: str = "") -> bool:
        if status not in {
            "completed",
            "failed",
            "waiting_input",
            "waiting_approval",
            "skipped",
            "cancelled",
            "unknown_outcome",
        }:
            raise ValueError("invalid schedule run status")
        with self._connection() as db:
            cursor = db.execute(
                "UPDATE eurus_schedule_runs SET status=?, finished_at=?, result=? "
                "WHERE run_id=? AND status IN ('queued', 'running')",
                (status, _iso(_utcnow()), result[:4000], run_id),
            )
            return cursor.rowcount == 1

    def history(self, schedule_id: str, *, limit: int = 20) -> list[ScheduleRun]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        with self._connection() as db:
            rows = db.execute(
                "SELECT * FROM eurus_schedule_runs WHERE schedule_id=? "
                "ORDER BY started_at DESC LIMIT ?",
                (schedule_id, limit),
            ).fetchall()
        return [
            ScheduleRun(
                str(row["run_id"]),
                str(row["schedule_id"]),
                str(row["scheduled_at"]),
                str(row["status"]),
                str(row["started_at"]),
                row["finished_at"],
                str(row["result"]),
            )
            for row in rows
        ]

    @staticmethod
    def _schedule(row: sqlite3.Row) -> Schedule:
        return Schedule(
            str(row["schedule_id"]),
            str(row["owner_id"]),
            str(row["objective"]),
            row["kind"],
            str(row["timezone"]),
            str(row["next_run_at"]),
            row["interval_seconds"],
            bool(row["enabled"]),
            json.loads(row["profile_json"]),
            str(row["created_at"]),
            str(row["updated_at"]),
        )

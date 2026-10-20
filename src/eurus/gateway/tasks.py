from __future__ import annotations

import json
import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

TaskStatus = Literal[
    "queued",
    "running",
    "waiting_approval",
    "waiting_input",
    "paused",
    "recovering",
    "completed",
    "failed",
    "cancelled",
]
TRANSITIONS: dict[str, set[str]] = {
    "queued": {"running", "cancelled", "paused"},
    "running": {
        "waiting_approval",
        "waiting_input",
        "paused",
        "recovering",
        "completed",
        "failed",
        "cancelled",
    },
    "waiting_approval": {"queued", "running", "paused", "cancelled", "failed"},
    "waiting_input": {"queued", "running", "paused", "cancelled", "failed"},
    "paused": {"queued", "cancelled"},
    "recovering": {"running", "paused", "failed", "cancelled"},
    "completed": set(),
    "failed": {"queued"},
    "cancelled": set(),
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class DurableTask:
    task_id: str
    owner_id: str
    session_id: str
    objective: str
    status: TaskStatus
    state: dict[str, Any]
    created_at: str
    updated_at: str
    error: str = ""


class TaskStore:
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
                CREATE TABLE IF NOT EXISTS gateway_tasks (
                    task_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, objective TEXT NOT NULL,
                    status TEXT NOT NULL, state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    error TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS gateway_tasks_owner_updated
                    ON gateway_tasks(owner_id, updated_at);
                CREATE TABLE IF NOT EXISTS gateway_sessions (
                    user_id TEXT NOT NULL, chat_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL, PRIMARY KEY(user_id, chat_id)
                );
                CREATE TABLE IF NOT EXISTS gateway_chat_locks (
                    user_id TEXT NOT NULL, chat_id TEXT NOT NULL, locked INTEGER NOT NULL,
                    updated_at TEXT NOT NULL, PRIMARY KEY(user_id, chat_id)
                );
                CREATE TABLE IF NOT EXISTS gateway_updates (
                    platform TEXT NOT NULL, update_id TEXT NOT NULL,
                    received_at TEXT NOT NULL, PRIMARY KEY(platform, update_id)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=10000")
        return db

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        db = self._connect()
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def create(
        self,
        owner_id: str,
        session_id: str,
        objective: str,
        state: dict[str, Any] | None = None,
    ) -> DurableTask:
        if not owner_id or not session_id or not objective.strip():
            raise ValueError("owner, session, and objective are required")
        now = _now()
        task = DurableTask(
            uuid.uuid4().hex, owner_id, session_id, objective, "queued", state or {}, now, now
        )
        with self._connection() as db:
            db.execute(
                "INSERT INTO gateway_tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    task.task_id,
                    task.owner_id,
                    task.session_id,
                    task.objective,
                    task.status,
                    json.dumps(task.state, ensure_ascii=False),
                    now,
                    now,
                    "",
                ),
            )
        return task

    def get(self, task_id: str, *, owner_id: str | None = None) -> DurableTask | None:
        with self._connection() as db:
            if owner_id is None:
                row = db.execute(
                    "SELECT * FROM gateway_tasks WHERE task_id=?", (task_id,)
                ).fetchone()
            else:
                row = db.execute(
                    "SELECT * FROM gateway_tasks WHERE task_id=? AND owner_id=?",
                    (task_id, owner_id),
                ).fetchone()
        return self._row(row) if row else None

    def list_owner(self, owner_id: str, limit: int = 20) -> list[DurableTask]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        with self._connection() as db:
            rows = db.execute(
                "SELECT * FROM gateway_tasks WHERE owner_id=? ORDER BY updated_at DESC LIMIT ?",
                (owner_id, limit),
            ).fetchall()
        return [self._row(row) for row in rows]

    def transition(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        state: dict[str, Any] | None = None,
        error: str | None = None,
        owner_id: str | None = None,
    ) -> DurableTask:
        with self._connection() as db:
            query = "SELECT * FROM gateway_tasks WHERE task_id=?"
            args: tuple[str, ...] = (task_id,)
            if owner_id is not None:
                query += " AND owner_id=?"
                args = (task_id, owner_id)
            row = db.execute(query, args).fetchone()
            if row is None:
                raise KeyError("task not found")
            current = str(row["status"])
            if status != current and status not in TRANSITIONS[current]:
                raise ValueError(f"invalid task transition: {current} -> {status}")
            new_state = state if state is not None else json.loads(row["state_json"])
            new_error = str(row["error"]) if error is None else error
            db.execute(
                "UPDATE gateway_tasks SET status=?, state_json=?, updated_at=?, error=? "
                "WHERE task_id=?",
                (status, json.dumps(new_state, ensure_ascii=False), _now(), new_error, task_id),
            )
            updated = db.execute(
                "SELECT * FROM gateway_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        return self._row(updated)

    def current_session(self, user_id: str, chat_id: str) -> str:
        with self._connection() as db:
            row = db.execute(
                "SELECT session_id FROM gateway_sessions WHERE user_id=? AND chat_id=?",
                (user_id, chat_id),
            ).fetchone()
            if row is not None:
                return str(row["session_id"])
            session_id = f"telegram:{user_id}:{chat_id}"
            db.execute(
                "INSERT OR IGNORE INTO gateway_sessions "
                "(user_id, chat_id, session_id, updated_at) VALUES (?, ?, ?, ?)",
                (user_id, chat_id, session_id, _now()),
            )
            row = db.execute(
                "SELECT session_id FROM gateway_sessions WHERE user_id=? AND chat_id=?",
                (user_id, chat_id),
            ).fetchone()
            return str(row["session_id"])

    def reset_session(self, user_id: str, chat_id: str) -> str:
        session_id = f"telegram:{user_id}:{chat_id}:{uuid.uuid4().hex[:12]}"
        with self._connection() as db:
            db.execute(
                "INSERT INTO gateway_sessions "
                "(user_id, chat_id, session_id, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(user_id, chat_id) DO UPDATE SET "
                "session_id=excluded.session_id, updated_at=excluded.updated_at",
                (user_id, chat_id, session_id, _now()),
            )
        return session_id

    def set_chat_locked(self, user_id: str, chat_id: str, locked: bool) -> None:
        with self._connection() as db:
            db.execute(
                "INSERT INTO gateway_chat_locks "
                "(user_id, chat_id, locked, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(user_id, chat_id) DO UPDATE SET "
                "locked=excluded.locked, updated_at=excluded.updated_at",
                (user_id, chat_id, int(locked), _now()),
            )

    def is_chat_locked(self, user_id: str, chat_id: str) -> bool:
        with self._connection() as db:
            row = db.execute(
                "SELECT locked FROM gateway_chat_locks WHERE user_id=? AND chat_id=?",
                (user_id, chat_id),
            ).fetchone()
        return bool(row["locked"]) if row else False

    def claim_update(self, platform: str, update_id: str) -> bool:
        try:
            with self._connection() as db:
                db.execute(
                    "INSERT INTO gateway_updates(platform, update_id, received_at) "
                    "VALUES (?, ?, ?)",
                    (platform, update_id, _now()),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def pause_interrupted(self) -> int:
        now = _now()
        with self._connection() as db:
            cursor = db.execute(
                "UPDATE gateway_tasks SET status='paused', updated_at=?, "
                "error='process interrupted; review before resubmitting' "
                "WHERE status IN ('queued','running','recovering')",
                (now,),
            )
        return cursor.rowcount

    def recoverable(self) -> list[DurableTask]:
        with self._connection() as db:
            rows = db.execute(
                "SELECT * FROM gateway_tasks WHERE status IN "
                "('queued','running','recovering') ORDER BY created_at"
            ).fetchall()
        return [self._row(row) for row in rows]

    @staticmethod
    def _row(row: sqlite3.Row) -> DurableTask:
        return DurableTask(
            task_id=str(row["task_id"]),
            owner_id=str(row["owner_id"]),
            session_id=str(row["session_id"]),
            objective=str(row["objective"]),
            status=row["status"],
            state=json.loads(row["state_json"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            error=str(row["error"]),
        )

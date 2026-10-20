from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

_SECRET = re.compile(
    r"(?i)\b("
    r"sk-[A-Za-z0-9_-]{8,}|gsk_[A-Za-z0-9_-]{8,}|nvapi-[A-Za-z0-9_-]{8,}|"
    r"AIza[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9_-]{8,}|github_pat_[A-Za-z0-9_-]{8,}"
    r")\b"
)
_SENSITIVE_KEYS = {
    "authorization",
    "api_key",
    "apikey",
    "token",
    "password",
    "secret",
    "credential",
    "bot_token",
}


def redact(value: Any, key: str = "") -> Any:
    if key.lower() in _SENSITIVE_KEYS or any(
        part in key.lower() for part in ("api_key", "password", "secret", "token")
    ):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(name): redact(item, str(name)) for name, item in value.items()}
    if isinstance(value, list | tuple):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _SECRET.sub("[REDACTED]", value)[:20000]
    if value is None or isinstance(value, bool | int | float):
        return value
    return str(value)[:20000]


class TrajectoryStore:
    def __init__(self, path: Path, *, retention_days: int = 90) -> None:
        if retention_days < 1 or retention_days > 3650:
            raise ValueError("retention_days must be between 1 and 3650")
        self.path = path.expanduser()
        self.retention_days = retention_days
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.touch(mode=0o600, exist_ok=True)
        with suppress(OSError):
            os.chmod(self.path, 0o600)
        with self._connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS trajectory_events ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, session_id TEXT, "
                "step_id TEXT, event_type TEXT NOT NULL, outcome TEXT NOT NULL, "
                "payload TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS trajectory_task_time "
                "ON trajectory_events(task_id, created_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS trajectory_type_time "
                "ON trajectory_events(event_type, created_at)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def record(
        self,
        task_id: str,
        event_type: str,
        *,
        session_id: str | None = None,
        step_id: str | None = None,
        outcome: str = "recorded",
        payload: dict[str, Any] | None = None,
    ) -> int:
        if not task_id.strip() or len(task_id) > 200:
            raise ValueError("task_id must be non-empty and at most 200 characters")
        if not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", event_type):
            raise ValueError("invalid event_type")
        if len(outcome) > 80:
            raise ValueError("outcome is too long")
        clean = json.dumps(redact(payload or {}), ensure_ascii=False, sort_keys=True)
        with self._connection() as connection:
            cursor = connection.execute(
                "INSERT INTO trajectory_events("
                "task_id, session_id, step_id, event_type, outcome, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    session_id,
                    step_id,
                    event_type,
                    outcome,
                    clean,
                    datetime.now(UTC).isoformat(),
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return an inserted row ID")
            return cursor.lastrowid

    def query(
        self,
        *,
        task_id: str | None = None,
        event_type: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        clauses: list[str] = []
        params: list[Any] = []
        if task_id is not None:
            clauses.append("task_id=?")
            params.append(task_id)
        if event_type is not None:
            clauses.append("event_type=?")
            params.append(event_type)
        if since is not None:
            if since.tzinfo is None:
                raise ValueError("since must be timezone-aware")
            clauses.append("created_at>=?")
            params.append(since.astimezone(UTC).isoformat())
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id, task_id, session_id, step_id, event_type, outcome, payload, created_at "
                f"FROM trajectory_events{where} ORDER BY id DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        result = [dict(row) for row in rows]
        for row in result:
            row["payload"] = json.loads(row["payload"])
        return result

    def prune(self, *, now: datetime | None = None) -> int:
        current = now or datetime.now(UTC)
        if current.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        cutoff = (current.astimezone(UTC) - timedelta(days=self.retention_days)).isoformat()
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM trajectory_events WHERE created_at < ?", (cutoff,)
            )
            return cursor.rowcount

    def export_jsonl(self, *, task_id: str | None = None, limit: int = 1000) -> str:
        rows = self.query(task_id=task_id, limit=limit)
        return "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in reversed(rows)
        )

    def delete_task(self, task_id: str) -> int:
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM trajectory_events WHERE task_id=?", (task_id,))
            return cursor.rowcount


class TrajectoryRecorder:
    def __init__(self, store: TrajectoryStore, task_id: str, session_id: str | None = None) -> None:
        if not task_id.strip():
            raise ValueError("task_id must be non-empty")
        self.store = store
        self.task_id = task_id
        self.session_id = session_id

    def record_runtime_event(self, event: Any) -> int:
        kind = str(getattr(event, "kind", getattr(event, "event_type", "unknown")))
        payload = getattr(event, "payload", {})
        if not isinstance(payload, dict):
            payload = {"value": str(payload)}
        run_id = getattr(event, "run_id", None)
        if run_id is not None:
            payload = {**payload, "run_id": str(run_id)}
        status = getattr(event, "status", None)
        if status is not None:
            payload = {**payload, "status": str(getattr(status, "value", status))}
        return self.store.record(
            self.task_id,
            "runtime." + re.sub(r"[^a-z0-9_.-]", "_", kind.lower())[:55].strip("._-") or "unknown",
            session_id=self.session_id,
            step_id=str(getattr(event, "step_id", "")) or None,
            outcome=str(getattr(event, "outcome", "recorded"))[:80],
            payload=payload,
        )

    def record_agent_event(self, event: Any) -> int:
        kind = str(getattr(event, "kind", "unknown"))
        step = getattr(event, "step", 0)
        payload: dict[str, Any] = {"step": step}
        outcome = "recorded"
        tool_call = getattr(event, "tool_call", None)
        if tool_call is not None:
            payload["tool"] = getattr(tool_call, "name", "unknown")
            payload["arguments"] = getattr(tool_call, "arguments", {})
        tool_result = getattr(event, "tool_result", None)
        if tool_result is not None:
            payload["tool"] = getattr(tool_result, "name", "unknown")
            payload["is_error"] = bool(getattr(tool_result, "is_error", False))
            payload["result"] = str(getattr(tool_result, "output", ""))[:4000]
            outcome = "error" if payload["is_error"] else "success"
        text = getattr(event, "text", "")
        if text:
            payload["text"] = str(text)[:1000]
        usage = getattr(event, "usage", None)
        if usage is not None:
            payload["usage"] = {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        stop_reason = getattr(event, "stop_reason", "")
        if stop_reason:
            payload["stop_reason"] = stop_reason
            outcome = str(stop_reason)[:80]
        return self.store.record(
            self.task_id,
            f"agent.{kind}",
            session_id=self.session_id,
            step_id=str(step) if step else None,
            outcome=outcome,
            payload=payload,
        )

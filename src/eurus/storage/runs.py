from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from typing import Any

from eurus.runtime.models import Run, RunStatus, utc_now
from eurus.storage.database import Database, DatabasePath

_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|secret|password|authorization|credential|private[_-]?key|"
    r"access[_-]?token|refresh[_-]?token|(?:^|[_-])(?:token|bearer)(?:$|[_-]))",
    re.IGNORECASE,
)


class ConcurrentRunUpdate(RuntimeError):
    pass


def _datetime_value(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse_datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _serialize(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
        allow_nan=False,
    )


def _run_from_row(row: sqlite3.Row) -> Run:
    return Run(
        run_id=row["run_id"],
        owner_id=row["owner_id"],
        objective=row["objective"],
        parent_run_id=row["parent_run_id"],
        session_id=row["session_id"],
        status=RunStatus(row["status"]),
        stop_reason=row["stop_reason"],
        budget=json.loads(row["budget_json"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        started_at=_parse_datetime(row["started_at"]),
        finished_at=_parse_datetime(row["finished_at"]),
        schema_version=row["schema_version"],
        checkpoint_version=row["checkpoint_version"],
        result_ref=row["result_ref"],
        status_version=row["status_version"],
    )


def _validate_checkpoint(value: Any, path: str = "state") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            if _SENSITIVE_KEY.search(key):
                raise ValueError(f"checkpoint contains a sensitive field at {path}.{key}")
            _validate_checkpoint(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_checkpoint(nested, f"{path}[{index}]")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"checkpoint contains unsupported value at {path}")


class RunStore:
    def __init__(self, database: Database | DatabasePath) -> None:
        self.database = database if isinstance(database, Database) else Database(database)

    def create(self, run: Run) -> Run:
        with self.database.connection() as connection:
            connection.execute(
                """INSERT INTO eurus_runs (
                    run_id, owner_id, objective, parent_run_id, session_id, status,
                    stop_reason, budget_json, created_at, updated_at, started_at,
                    finished_at, schema_version, checkpoint_version, result_ref, status_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                self._values(run),
            )
        return run

    def get(self, run_id: str) -> Run | None:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT * FROM eurus_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return _run_from_row(row) if row is not None else None

    def list_runs(
        self,
        *,
        owner_id: str | None = None,
        parent_run_id: str | None = None,
        statuses: tuple[RunStatus, ...] = (),
        limit: int = 100,
    ) -> list[Run]:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        clauses: list[str] = []
        values: list[Any] = []
        if owner_id is not None:
            clauses.append("owner_id = ?")
            values.append(owner_id)
        if parent_run_id is not None:
            clauses.append("parent_run_id = ?")
            values.append(parent_run_id)
        if statuses:
            clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
            values.extend(status.value for status in statuses)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.database.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM eurus_runs{where} ORDER BY created_at DESC LIMIT ?",
                (*values, limit),
            ).fetchall()
        return [_run_from_row(row) for row in rows]

    def save(self, run: Run, *, expected_status_version: int | None = None) -> Run:
        values = self._values(run)
        columns = (
            "owner_id = ?, objective = ?, parent_run_id = ?, session_id = ?, status = ?, "
            "stop_reason = ?, budget_json = ?, created_at = ?, updated_at = ?, started_at = ?, "
            "finished_at = ?, schema_version = ?, result_ref = ?, status_version = ?"
        )
        parameters = (*values[1:13], *values[14:], run.run_id)
        query = f"UPDATE eurus_runs SET {columns} WHERE run_id = ?"
        if expected_status_version is not None:
            query += " AND status_version = ?"
            parameters = (*parameters, expected_status_version)
        with self.database.connection() as connection:
            cursor = connection.execute(query, parameters)
            if cursor.rowcount != 1:
                if self.get(run.run_id) is None:
                    raise KeyError(f"run {run.run_id} does not exist")
                raise ConcurrentRunUpdate(f"run {run.run_id} changed since it was read")
        return run

    def request_cancel(self, run_id: str) -> bool:
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM eurus_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None or row["status"] != RunStatus.RUNNING.value:
                connection.rollback()
                return False
            connection.execute(
                "INSERT INTO eurus_run_controls(run_id, cancel_requested, updated_at) "
                "VALUES (?, 1, ?) ON CONFLICT(run_id) DO UPDATE SET "
                "cancel_requested = 1, updated_at = excluded.updated_at",
                (run_id, utc_now().isoformat()),
            )
            connection.commit()
        return True

    def cancellation_requested(self, run_id: str) -> bool:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM eurus_run_controls WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return row is not None and bool(row["cancel_requested"])

    def save_checkpoint(
        self,
        run_id: str,
        state: dict[str, Any],
        *,
        checkpoint_type: str = "runtime",
        expected_version: int | None = None,
    ) -> int:
        if not checkpoint_type.strip():
            raise ValueError("checkpoint_type must be non-empty")
        _validate_checkpoint(state)
        state_json = _serialize(state)
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT checkpoint_version FROM eurus_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"run {run_id} does not exist")
            current_version = int(row["checkpoint_version"])
            if expected_version is not None and current_version != expected_version:
                raise ConcurrentRunUpdate(f"checkpoint for run {run_id} changed since it was read")
            next_version = current_version + 1
            connection.execute(
                """INSERT INTO eurus_run_checkpoints(
                    run_id, checkpoint_version, checkpoint_type, state_json, created_at
                ) VALUES (?, ?, ?, ?, ?)""",
                (run_id, next_version, checkpoint_type, state_json, utc_now().isoformat()),
            )
            connection.execute(
                "UPDATE eurus_runs SET checkpoint_version = ?, updated_at = ? WHERE run_id = ?",
                (next_version, utc_now().isoformat(), run_id),
            )
            connection.commit()
        return next_version

    def latest_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        with self.database.connection() as connection:
            row = connection.execute(
                """SELECT checkpoint_version, checkpoint_type, state_json, created_at
                FROM eurus_run_checkpoints WHERE run_id = ?
                ORDER BY checkpoint_version DESC LIMIT 1""",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "run_id": run_id,
            "checkpoint_version": int(row["checkpoint_version"]),
            "checkpoint_type": row["checkpoint_type"],
            "state": json.loads(row["state_json"]),
            "created_at": datetime.fromisoformat(row["created_at"]),
        }

    @staticmethod
    def _values(run: Run) -> tuple[Any, ...]:
        return (
            run.run_id,
            run.owner_id,
            run.objective,
            run.parent_run_id,
            run.session_id,
            run.status.value,
            run.stop_reason,
            _serialize(run.budget),
            run.created_at.isoformat(),
            run.updated_at.isoformat(),
            _datetime_value(run.started_at),
            _datetime_value(run.finished_at),
            run.schema_version,
            run.checkpoint_version,
            run.result_ref,
            run.status_version,
        )

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from eurus.runtime.models import RuntimeEvent
from eurus.storage.database import Database, DatabasePath


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


def _event_from_row(row: sqlite3.Row) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=row["event_id"],
        run_id=row["run_id"],
        event_type=row["event_type"],
        sequence_no=row["sequence_no"],
        payload=json.loads(row["payload_json"]),
        payload_version=row["payload_version"],
        created_at=datetime.fromisoformat(row["created_at"]),
        step_id=row["step_id"],
        attempt_id=row["attempt_id"],
        parent_run_id=row["parent_run_id"],
    )


class EventStore:
    def __init__(self, database: Database | DatabasePath) -> None:
        self.database = database if isinstance(database, Database) else Database(database)

    def last_sequence(self, run_id: str) -> int:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT MAX(sequence_no) AS sequence_no FROM eurus_runtime_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return 0
        return int(row["sequence_no"] or 0)

    def append(self, event: RuntimeEvent) -> RuntimeEvent:
        with self.database.connection() as connection:
            connection.execute(
                """INSERT INTO eurus_runtime_events (
                    event_id, run_id, sequence_no, event_type, payload_json,
                    payload_version, created_at, step_id, attempt_id, parent_run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.event_id,
                    event.run_id,
                    event.sequence_no,
                    event.event_type,
                    _serialize(event.payload),
                    event.payload_version,
                    event.created_at.isoformat(),
                    event.step_id,
                    event.attempt_id,
                    event.parent_run_id,
                ),
            )
        return event

    def list_for_run(
        self, run_id: str, *, after_sequence: int = 0, limit: int = 1000
    ) -> list[RuntimeEvent]:
        if after_sequence < 0 or limit < 1 or limit > 10000:
            raise ValueError(
                "after_sequence must be non-negative and limit must be between 1 and 10000"
            )
        with self.database.connection() as connection:
            rows = connection.execute(
                """SELECT * FROM eurus_runtime_events
                WHERE run_id = ? AND sequence_no > ? ORDER BY sequence_no LIMIT ?""",
                (run_id, after_sequence, limit),
            ).fetchall()
        return [_event_from_row(row) for row in rows]

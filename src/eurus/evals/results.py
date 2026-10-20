from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    suite: str
    revision: str
    task: str
    passed: bool
    steps: int
    tokens: int
    seconds: float
    stop_reason: str
    detail: str = ""
    metrics: dict[str, int | float | str | bool] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvaluationStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with closing(sqlite3.connect(self.path, timeout=10)) as connection, connection:
            connection.execute("PRAGMA busy_timeout=10000")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS evaluation_results ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, suite TEXT NOT NULL, "
                "revision TEXT NOT NULL, task TEXT NOT NULL, passed INTEGER NOT NULL, "
                "steps INTEGER NOT NULL, tokens INTEGER NOT NULL, seconds REAL NOT NULL, "
                "stop_reason TEXT NOT NULL, detail TEXT NOT NULL, metrics TEXT NOT NULL, "
                "created_at TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS evaluation_suite_revision "
                "ON evaluation_results(suite, revision, created_at)"
            )

    def record(self, result: EvaluationRecord) -> int:
        if not result.suite.strip() or not result.revision.strip() or not result.task.strip():
            raise ValueError("suite, revision, and task are required")
        if result.steps < 0 or result.tokens < 0 or result.seconds < 0:
            raise ValueError("evaluation measurements cannot be negative")
        with closing(sqlite3.connect(self.path, timeout=10)) as connection, connection:
            connection.execute("PRAGMA busy_timeout=10000")
            cursor = connection.execute(
                "INSERT INTO evaluation_results(suite, revision, task, passed, steps, tokens, "
                "seconds, stop_reason, detail, metrics, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result.suite,
                    result.revision,
                    result.task,
                    int(result.passed),
                    result.steps,
                    result.tokens,
                    result.seconds,
                    result.stop_reason,
                    result.detail[:2000],
                    json.dumps(result.metrics or {}, sort_keys=True),
                    datetime.now(UTC).isoformat(),
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return an inserted row ID")
            return cursor.lastrowid

    def query(
        self, *, suite: str | None = None, revision: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        clauses: list[str] = []
        params: list[Any] = []
        if suite is not None:
            clauses.append("suite=?")
            params.append(suite)
        if revision is not None:
            clauses.append("revision=?")
            params.append(revision)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with closing(sqlite3.connect(self.path, timeout=10)) as connection, connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM evaluation_results" + where + " ORDER BY id DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        result = [dict(row) for row in rows]
        for item in result:
            item["passed"] = bool(item["passed"])
            item["metrics"] = json.loads(item["metrics"])
        return result

    def compare(
        self, suite: str, baseline_revision: str, candidate_revision: str
    ) -> dict[str, Any]:
        baseline = self.query(suite=suite, revision=baseline_revision, limit=1000)
        candidate = self.query(suite=suite, revision=candidate_revision, limit=1000)

        def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "runs": len(rows),
                "passed": sum(row["passed"] for row in rows),
                "pass_rate": sum(row["passed"] for row in rows) / len(rows) if rows else None,
                "mean_seconds": sum(row["seconds"] for row in rows) / len(rows) if rows else None,
                "mean_steps": sum(row["steps"] for row in rows) / len(rows) if rows else None,
                "mean_tokens": sum(row["tokens"] for row in rows) / len(rows) if rows else None,
            }

        return {"suite": suite, "baseline": summarize(baseline), "candidate": summarize(candidate)}

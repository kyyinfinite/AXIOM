from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from eurus.models import Message


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SessionStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.touch(mode=0o600, exist_ok=True)
        with suppress(OSError):
            os.chmod(self.path, 0o600)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
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

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS session_summaries (
                    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
                    summary TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS session_lineage (
                    child_id TEXT PRIMARY KEY,
                    parent_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS session_lineage_parent ON session_lineage(parent_id);
                CREATE TABLE IF NOT EXISTS session_message_search (
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    project_scope TEXT NOT NULL DEFAULT 'global',
                    PRIMARY KEY (session_id, sequence)
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS session_messages_fts USING fts5(
                    session_id UNINDEXED, sequence UNINDEXED, role UNINDEXED,
                    project_scope UNINDEXED, content
                );
                CREATE TABLE IF NOT EXISTS lifecycle_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    event_type TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS lifecycle_events_type_created
                    ON lifecycle_events(event_type, created_at);
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '',
                    category TEXT NOT NULL DEFAULT 'general',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT '',
                    memory_type TEXT NOT NULL DEFAULT 'fact',
                    confidence REAL NOT NULL DEFAULT 0.7,
                    source TEXT NOT NULL DEFAULT 'user',
                    use_count INTEGER NOT NULL DEFAULT 0,
                    last_used_at TEXT NOT NULL DEFAULT ''
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content, tags, content='memories', content_rowid='id'
                );
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content, tags)
                    VALUES (new.id, new.content, new.tags);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                    VALUES ('delete', old.id, old.content, old.tags);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
                    VALUES ('delete', old.id, old.content, old.tags);
                    INSERT INTO memories_fts(rowid, content, tags)
                    VALUES (new.id, new.content, new.tags);
                END;
                """
            )

            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(memories)").fetchall()
            }
            if "category" not in columns:
                connection.execute(
                    "ALTER TABLE memories ADD COLUMN category TEXT NOT NULL DEFAULT 'general'"
                )
            migrations = {
                "updated_at": "TEXT NOT NULL DEFAULT ''",
                "memory_type": "TEXT NOT NULL DEFAULT 'fact'",
                "confidence": "REAL NOT NULL DEFAULT 0.7",
                "source": "TEXT NOT NULL DEFAULT 'user'",
                "use_count": "INTEGER NOT NULL DEFAULT 0",
                "last_used_at": "TEXT NOT NULL DEFAULT ''",
            }
            for name, definition in migrations.items():
                if name not in columns:
                    connection.execute(f"ALTER TABLE memories ADD COLUMN {name} {definition}")
            if "scope" not in columns:
                connection.execute(
                    "ALTER TABLE memories ADD COLUMN scope TEXT NOT NULL DEFAULT 'global'"
                )
            connection.execute("UPDATE memories SET updated_at = created_at WHERE updated_at = ''")
            indexed = connection.execute("SELECT COUNT(*) FROM session_message_search").fetchone()[
                0
            ]
            stored = connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            if indexed == 0 and stored > 0:
                connection.execute("DELETE FROM session_messages_fts")
                connection.execute("DELETE FROM session_message_search")
                rows = connection.execute(
                    "SELECT session_id, sequence, payload FROM messages "
                    "ORDER BY session_id, sequence"
                ).fetchall()
                for row in rows:
                    try:
                        payload = json.loads(row["payload"])
                    except (TypeError, json.JSONDecodeError):
                        continue
                    content = str(payload.get("content", "")).strip()
                    if not content:
                        continue
                    values = (
                        str(row["session_id"]),
                        int(row["sequence"]),
                        str(payload.get("role", "")),
                        content,
                        "global",
                    )
                    connection.execute(
                        "INSERT OR REPLACE INTO session_message_search("
                        "session_id, sequence, role, content, project_scope) "
                        "VALUES (?, ?, ?, ?, ?)",
                        values,
                    )
                    connection.execute(
                        "INSERT INTO session_messages_fts("
                        "session_id, sequence, role, project_scope, content) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (values[0], values[1], values[2], values[4], values[3]),
                    )

    def save_messages(
        self, session_id: str, messages: list[Message], *, project_scope: str = "global"
    ) -> None:
        if not session_id.strip():
            raise ValueError("session id must not be empty")
        persisted = [message for message in messages if message.role != "system"]
        now = _now()
        title = next(
            (
                message.content.strip().splitlines()[0][:80]
                for message in persisted
                if message.role == "user" and message.content.strip()
            ),
            session_id,
        )
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO sessions(id, title, created_at, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET title = ?, updated_at = ?""",
                (session_id, title, now, now, title, now),
            )
            connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            connection.execute(
                "DELETE FROM session_message_search WHERE session_id = ?", (session_id,)
            )
            connection.execute(
                "DELETE FROM session_messages_fts WHERE session_id = ?", (session_id,)
            )
            connection.executemany(
                "INSERT INTO messages(session_id, sequence, payload, created_at) "
                "VALUES (?, ?, ?, ?)",
                [
                    (session_id, index, json.dumps(message.to_dict(), ensure_ascii=False), now)
                    for index, message in enumerate(persisted)
                ],
            )
            searchable = [
                (session_id, index, message.role, message.content.strip(), project_scope)
                for index, message in enumerate(persisted)
                if message.content.strip() and message.role in {"user", "assistant", "tool"}
            ]
            connection.executemany(
                "INSERT INTO session_message_search("
                "session_id, sequence, role, content, project_scope) "
                "VALUES (?, ?, ?, ?, ?)",
                searchable,
            )
            connection.executemany(
                "INSERT INTO session_messages_fts("
                "session_id, sequence, role, project_scope, content) "
                "VALUES (?, ?, ?, ?, ?)",
                [(row[0], row[1], row[2], row[4], row[3]) for row in searchable],
            )

    def load_messages(self, session_id: str) -> list[Message]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT payload FROM messages WHERE session_id = ? ORDER BY sequence",
                (session_id,),
            ).fetchall()
        return [Message.from_dict(json.loads(row["payload"])) for row in rows]

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT s.id, s.title, s.created_at, s.updated_at,
                          COUNT(m.sequence) AS message_count
                   FROM sessions s LEFT JOIN messages m ON m.session_id = s.id
                   GROUP BY s.id ORDER BY s.updated_at DESC"""
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_session(self, session_id: str) -> bool:
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM session_messages_fts WHERE session_id = ?", (session_id,)
            )
            connection.execute(
                "DELETE FROM session_message_search WHERE session_id = ?", (session_id,)
            )
            connection.execute(
                "DELETE FROM session_lineage WHERE child_id = ? OR parent_id = ?",
                (session_id, session_id),
            )
            connection.execute("DELETE FROM session_summaries WHERE session_id = ?", (session_id,))
            cursor = connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0

    def search_session_messages(
        self, query: str, *, limit: int = 10, project_scope: str | None = None
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")
        terms = re.findall(r"[\w]+", query.strip(), flags=re.UNICODE)
        if not terms:
            return []
        match_query = " AND ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms[:12])
        sql = (
            "SELECT s.session_id, s.sequence, s.role, s.content, s.project_scope, "
            "ss.title FROM session_messages_fts f "
            "JOIN session_message_search s ON s.session_id = f.session_id "
            "AND s.sequence = f.sequence "
            "LEFT JOIN sessions ss ON ss.id = s.session_id WHERE session_messages_fts MATCH ?"
        )
        params: list[Any] = [match_query]
        if project_scope:
            sql += " AND s.project_scope IN ('global', ?)"
            params.append(project_scope)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        with self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def export_privacy_data(self) -> dict[str, Any]:
        with self._connection() as connection:
            sessions = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM sessions ORDER BY created_at"
                ).fetchall()
            ]
            messages = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM messages ORDER BY session_id, sequence"
                ).fetchall()
            ]
            memories = [
                dict(row)
                for row in connection.execute("SELECT * FROM memories ORDER BY id").fetchall()
            ]
            summaries = [
                dict(row)
                for row in connection.execute("SELECT * FROM session_summaries").fetchall()
            ]
            lineage = [
                dict(row) for row in connection.execute("SELECT * FROM session_lineage").fetchall()
            ]
        return {
            "sessions": sessions,
            "messages": messages,
            "memories": memories,
            "session_summaries": summaries,
            "session_lineage": lineage,
        }

    def purge_privacy_data(self) -> dict[str, int]:
        with self._connection() as connection:
            counts = {
                "sessions": int(connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]),
                "messages": int(connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]),
                "memories": int(connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]),
                "events": int(
                    connection.execute("SELECT COUNT(*) FROM lifecycle_events").fetchone()[0]
                ),
            }
            connection.execute("DELETE FROM session_messages_fts")
            connection.execute("DELETE FROM session_message_search")
            connection.execute("DELETE FROM messages")
            connection.execute("DELETE FROM session_summaries")
            connection.execute("DELETE FROM session_lineage")
            connection.execute("DELETE FROM sessions")
            connection.execute("DELETE FROM memories")
            connection.execute("DELETE FROM lifecycle_events")
            connection.execute("INSERT INTO memories_fts(memories_fts) VALUES ('rebuild')")
        return counts

    def record_event(
        self,
        event_type: str,
        *,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", event_type):
            raise ValueError("invalid event type")
        payload = json.dumps(metadata or {}, ensure_ascii=False, separators=(",", ":"))
        if len(payload) > 2_000:
            raise ValueError("event metadata exceeds 2000 characters")
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO lifecycle_events(session_id, event_type, metadata, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, event_type, payload, _now()),
            )

    def event_metrics(self, *, days: int | None = None) -> dict[str, int]:
        if days is not None and not 1 <= days <= 3650:
            raise ValueError("days must be between 1 and 3650")
        query = (
            "SELECT event_type, COUNT(*) AS total FROM lifecycle_events "
            "{where} GROUP BY event_type ORDER BY event_type"
        )
        if days is None:
            sql = query.format(where="")
            params: tuple[str, ...] = ()
        else:
            cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
            sql = query.format(where="WHERE created_at >= ?")
            params = (cutoff,)
        with self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        return {str(row["event_type"]): int(row["total"]) for row in rows}

    def event_rates(self, *, days: int = 30) -> dict[str, float | int]:
        counts = self.event_metrics(days=days)
        approval_decisions = counts.get("approval_accepted", 0) + counts.get("approval_rejected", 0)
        correction_decisions = counts.get("correction_accepted", 0) + counts.get(
            "correction_rejected", 0
        )
        return {
            "days": days,
            "approval_decisions": approval_decisions,
            "approval_rejection_rate": (
                counts.get("approval_rejected", 0) / approval_decisions
                if approval_decisions
                else 0.0
            ),
            "correction_proposals": counts.get("correction_proposed", 0),
            "correction_decisions": correction_decisions,
            "correction_acceptance_rate": (
                counts.get("correction_accepted", 0) / correction_decisions
                if correction_decisions
                else 0.0
            ),
            "correction_rejection_rate": (
                counts.get("correction_rejected", 0) / correction_decisions
                if correction_decisions
                else 0.0
            ),
        }

    def build_session_summary(
        self, session_id: str, *, max_entries: int = 20, max_chars: int = 6_000
    ) -> str:
        if not 1 <= max_entries <= 100 or not 100 <= max_chars <= 20_000:
            raise ValueError("invalid session summary limits")
        messages = self.load_messages(session_id)
        entries: list[str] = []
        for message in messages:
            content = " ".join(message.content.split())
            if message.role in {"user", "assistant"} and content:
                label = "User" if message.role == "user" else "Agent"
                entries.append(f"- {label}: {content[:220]}")
        summary = "\n".join(entries[-max_entries:])[:max_chars]
        if not summary:
            raise ValueError("session has no summarizable text")
        return summary

    def save_session_summary(self, session_id: str, summary: str) -> None:
        value = summary.strip()
        if not session_id.strip() or not value:
            raise ValueError("session id and summary must not be empty")
        if len(value) > 20_000:
            raise ValueError("session summary exceeds 20000 characters")
        now = _now()
        with self._connection() as connection:
            exists = connection.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if exists is None:
                raise ValueError("session does not exist")
            connection.execute(
                """INSERT INTO session_summaries(session_id, summary, updated_at)
                   VALUES (?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET
                   summary = excluded.summary, updated_at = excluded.updated_at""",
                (session_id, value, now),
            )

    def load_session_summary(self, session_id: str) -> str | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT summary FROM session_summaries WHERE session_id = ?", (session_id,)
            ).fetchone()
        return str(row["summary"]) if row else None

    def save_session_lineage(
        self, parent_id: str, child_id: str, reason: str, summary: str
    ) -> None:
        if not parent_id.strip() or not child_id.strip() or parent_id == child_id:
            raise ValueError("parent and child session ids must be distinct and non-empty")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", reason):
            raise ValueError("invalid lineage reason")
        value = summary.strip()
        if not value or len(value) > 20_000:
            raise ValueError("lineage summary must contain 1 to 20000 characters")
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO session_lineage(child_id, parent_id, reason, summary, created_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(child_id) DO UPDATE SET "
                "parent_id = excluded.parent_id, reason = excluded.reason, "
                "summary = excluded.summary, created_at = excluded.created_at",
                (child_id, parent_id, reason, value, _now()),
            )

    def session_lineage(self, child_id: str) -> dict[str, str] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT parent_id, child_id, reason, summary "
                "FROM session_lineage WHERE child_id = ?",
                (child_id,),
            ).fetchone()
        return (
            {key: str(row[key]) for key in ("parent_id", "child_id", "reason", "summary")}
            if row
            else None
        )

    def latest_session_descendant(self, session_id: str) -> str:
        current = session_id
        visited = {current}
        with self._connection() as connection:
            while True:
                row = connection.execute(
                    "SELECT child_id FROM session_lineage WHERE parent_id = ? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (current,),
                ).fetchone()
                if row is None or str(row["child_id"]) in visited:
                    return current
                child = str(row["child_id"])
                exists = connection.execute(
                    "SELECT 1 FROM sessions WHERE id = ?", (child,)
                ).fetchone()
                if exists is None:
                    return current
                visited.add(child)
                current = child

    def memory_curation_due(self, *, interval_days: int = 30) -> bool:
        if not 1 <= interval_days <= 365:
            raise ValueError("interval_days must be between 1 and 365")
        cutoff = (datetime.now(UTC) - timedelta(days=interval_days)).isoformat()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM lifecycle_events "
                "WHERE event_type = 'memory_curation_nudge' AND created_at >= ? LIMIT 1",
                (cutoff,),
            ).fetchone()
        return row is None

    def consolidate_memories(self, *, dry_run: bool = True, limit: int = 500) -> dict[str, int]:
        if not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id, content, memory_type, category FROM memories ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
            seen: set[tuple[str, str, str]] = set()
            duplicate_ids: list[int] = []
            for row in rows:
                normalized_content = " ".join(str(row["content"]).casefold().split())
                key = (str(row["memory_type"]), str(row["category"]), normalized_content)
                if key in seen:
                    duplicate_ids.append(int(row["id"]))
                else:
                    seen.add(key)
            if duplicate_ids and not dry_run:
                connection.executemany(
                    "DELETE FROM memories WHERE id = ?",
                    [(item,) for item in duplicate_ids],
                )
        deleted = 0 if dry_run else len(duplicate_ids)
        return {"scanned": len(rows), "duplicates": len(duplicate_ids), "deleted": deleted}

    def add_memory(
        self,
        content: str,
        tags: str = "",
        *,
        category: str = "general",
        memory_type: str = "fact",
        confidence: float = 0.7,
        source: str = "user",
        scope: str = "global",
    ) -> int:
        value = content.strip()
        category_value = category.strip().lower() or "general"
        type_value = memory_type.strip().lower() or "fact"
        if type_value not in {
            "profile",
            "preference",
            "project_fact",
            "agent_note",
            "lesson",
            "episode",
            "fact",
        }:
            raise ValueError("unsupported memory type")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not value:
            raise ValueError("memory content must not be empty")
        if len(value) > 20_000:
            raise ValueError("memory content exceeds 20000 characters")
        if len(category_value) > 64:
            raise ValueError("memory category exceeds 64 characters")
        now = _now()
        with self._connection() as connection:
            cursor = connection.execute(
                """INSERT INTO memories(
                       content, tags, category, created_at, updated_at, memory_type,
                       confidence, source, scope
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    value,
                    tags.strip(),
                    category_value,
                    now,
                    now,
                    type_value,
                    confidence,
                    source,
                    scope.strip() or "global",
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a memory ID")
            return int(cursor.lastrowid)

    def update_memory(
        self,
        memory_id: int,
        content: str,
        *,
        tags: str | None = None,
        category: str | None = None,
    ) -> bool:
        value = content.strip()
        if not value:
            raise ValueError("memory content must not be empty")
        if len(value) > 20_000:
            raise ValueError("memory content exceeds 20000 characters")
        if category is not None and len(category.strip()) > 64:
            raise ValueError("memory category exceeds 64 characters")
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT tags, category FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if existing is None:
                return False
            next_tags = existing["tags"] if tags is None else tags.strip()
            next_category = (
                existing["category"]
                if category is None
                else (category.strip().lower() or "general")
            )
            connection.execute(
                """UPDATE memories SET content = ?, tags = ?, category = ?, updated_at = ?
                   WHERE id = ?""",
                (value, next_tags, next_category, _now(), memory_id),
            )
        return True

    def delete_memory(self, memory_id: int) -> bool:
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return cursor.rowcount > 0

    def search_memories_any(
        self,
        terms: Sequence[str],
        limit: int = 5,
        *,
        category: str | None = None,
        scope: str | None = None,
    ) -> list[dict[str, Any]]:
        quoted = [f'"{term}"' for term in terms if term.isalnum()]
        if not quoted or limit < 1:
            return []
        sql = (
            "SELECT m.id, m.content, m.tags, m.category, m.created_at, m.updated_at, "
            "m.memory_type, m.confidence, m.source, m.use_count, m.last_used_at, m.scope "
            "FROM memories_fts f JOIN memories m ON m.id = f.rowid WHERE memories_fts MATCH ?"
        )
        params: list[Any] = [" OR ".join(quoted)]
        if category is not None:
            sql += " AND m.category = ?"
            params.append(category)
        if scope is not None:
            sql += " AND m.scope IN ('global', ?)"
            params.append(scope)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        with self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def list_memories(
        self, *, category: str | None = None, limit: int = 50, scope: str | None = None
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT id, content, tags, category, created_at, updated_at, memory_type, "
            "confidence, source, use_count, last_used_at, scope FROM memories"
        )
        params: list[Any] = []
        conditions: list[str] = []
        if category is not None:
            conditions.append("category = ?")
            params.append(category)
        if scope is not None:
            conditions.append("scope IN ('global', ?)")
            params.append(scope)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def relevant_memories(
        self, terms: Sequence[str], limit: int = 8, *, scope: str | None = None
    ) -> list[dict[str, Any]]:
        rows = self.search_memories_any(terms, limit, scope=scope)
        if not rows:
            return []
        ids = [int(row["id"]) for row in rows]
        with self._connection() as connection:
            connection.executemany(
                "UPDATE memories SET use_count = use_count + 1, last_used_at = ? WHERE id = ?",
                [(_now(), memory_id) for memory_id in ids],
            )
        return rows

    def search_memories(
        self, query: str, limit: int = 10, *, scope: str | None = None
    ) -> list[dict[str, Any]]:
        cleaned = query.strip()
        if not cleaned:
            return []
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        terms = re.findall(r"[\w]+", cleaned, flags=re.UNICODE)
        if not terms:
            return []
        match_query = " AND ".join(f'"{part}"' for part in terms)
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT m.id, m.content, m.tags, m.category, m.created_at, m.updated_at,
                          m.memory_type, m.confidence, m.source, m.use_count,
                          m.last_used_at, m.scope
                   FROM memories_fts f JOIN memories m ON m.id = f.rowid
                   WHERE memories_fts MATCH ? AND (? IS NULL OR m.scope IN ('global', ?))
                   ORDER BY rank LIMIT ?""",
                (match_query, scope, scope, limit),
            ).fetchall()
        return [dict(row) for row in rows]

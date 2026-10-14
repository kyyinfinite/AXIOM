from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from axiom.models import Message


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
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '',
                    category TEXT NOT NULL DEFAULT 'general',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT ''
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
            # Additive migrations keep databases created by original readable.
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(memories)").fetchall()
            }
            if "category" not in columns:
                connection.execute(
                    "ALTER TABLE memories ADD COLUMN category TEXT NOT NULL DEFAULT 'general'"
                )
            if "updated_at" not in columns:
                connection.execute(
                    "ALTER TABLE memories ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''"
                )
            connection.execute("UPDATE memories SET updated_at = created_at WHERE updated_at = ''")

    def save_messages(self, session_id: str, messages: list[Message]) -> None:
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
            connection.executemany(
                "INSERT INTO messages(session_id, sequence, payload, created_at) "
                "VALUES (?, ?, ?, ?)",
                [
                    (session_id, index, json.dumps(message.to_dict(), ensure_ascii=False), now)
                    for index, message in enumerate(persisted)
                ],
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
            cursor = connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0

    def add_memory(self, content: str, tags: str = "", *, category: str = "general") -> int:
        value = content.strip()
        category_value = category.strip().lower() or "general"
        if not value:
            raise ValueError("memory content must not be empty")
        if len(value) > 20_000:
            raise ValueError("memory content exceeds 20000 characters")
        if len(category_value) > 64:
            raise ValueError("memory category exceeds 64 characters")
        now = _now()
        with self._connection() as connection:
            cursor = connection.execute(
                """INSERT INTO memories(content, tags, category, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (value, tags.strip(), category_value, now, now),
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
        self, terms: Sequence[str], limit: int = 5, *, category: str | None = None
    ) -> list[dict[str, Any]]:
        quoted = [f'"{term}"' for term in terms if term.isalnum()]
        if not quoted or limit < 1:
            return []
        sql = (
            "SELECT m.id, m.content, m.tags, m.category, m.created_at, m.updated_at "
            "FROM memories_fts f JOIN memories m ON m.id = f.rowid WHERE memories_fts MATCH ?"
        )
        params: list[Any] = [" OR ".join(quoted)]
        if category is not None:
            sql += " AND m.category = ?"
            params.append(category)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        with self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def list_memories(
        self, *, category: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        sql = "SELECT id, content, tags, category, created_at, updated_at FROM memories"
        params: list[Any] = []
        if category is not None:
            sql += " WHERE category = ?"
            params.append(category)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def search_memories(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
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
                """SELECT m.id, m.content, m.tags, m.category, m.created_at, m.updated_at
                   FROM memories_fts f JOIN memories m ON m.id = f.rowid
                   WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?""",
                (match_query, limit),
            ).fetchall()
        return [dict(row) for row in rows]

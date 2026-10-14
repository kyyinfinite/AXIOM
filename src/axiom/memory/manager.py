from __future__ import annotations

from pathlib import Path
from typing import Any

from axiom.sessions import SessionStore


class MemoryManager:
    """Small facade over the original SQLite/FTS5 store."""

    def __init__(self, database: Path) -> None:
        self.store = SessionStore(database)

    def remember(self, content: str, *, category: str = "general", tags: str = "") -> int:
        return self.store.add_memory(content, tags, category=category)

    def recall(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        return self.store.search_memories(query, limit=limit)

    def forget(self, memory_id: int) -> bool:
        return self.store.delete_memory(memory_id)

    def update(
        self,
        memory_id: int,
        content: str,
        *,
        category: str | None = None,
        tags: str | None = None,
    ) -> bool:
        return self.store.update_memory(memory_id, content, category=category, tags=tags)

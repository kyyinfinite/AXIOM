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


class PromptMemory:
    def __init__(self, database: Path, lessons: Any, *, max_chars: int = 1800) -> None:
        self.database = database
        self.lessons = lessons
        self.max_chars = max_chars

    def relevant(self, task: str) -> list[str]:
        import re

        terms = list(
            dict.fromkeys(
                word
                for word in re.findall(r"\w+", task.lower())
                if len(word) >= 3 and word.isalnum()
            )
        )[:12]
        items = self.lessons.relevant(task)
        if self.database.exists() and terms:
            manager = MemoryManager(self.database)
            rows = manager.store.search_memories_any(terms, 8)
            items.extend(
                f"[{row.get('memory_type', 'fact')}] {row['content']}"
                for row in rows
                if row.get("category") != "lesson" and float(row.get("confidence", 0.7)) >= 0.55
            )
        result: list[str] = []
        used = 0
        for item in items:
            clean = str(item).strip()
            if not clean or used + len(clean) > self.max_chars:
                continue
            result.append(clean)
            used += len(clean)
        return result

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from eurus.sessions import SessionStore


class MemoryProvider(ABC):
    @abstractmethod
    def recall(
        self, query: str, *, limit: int = 5, scope: str | None = None
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def remember(
        self, content: str, *, category: str = "general", tags: str = "", scope: str = "global"
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def forget(self, memory_id: int) -> bool:
        raise NotImplementedError


class SQLiteMemoryProvider(MemoryProvider):
    def __init__(self, database: Path) -> None:
        self.store = SessionStore(database)

    def recall(
        self, query: str, *, limit: int = 5, scope: str | None = None
    ) -> list[dict[str, Any]]:
        return self.store.search_memories(query, limit=min(max(limit, 1), 100), scope=scope)

    def remember(
        self, content: str, *, category: str = "general", tags: str = "", scope: str = "global"
    ) -> int:
        return self.store.add_memory(content, tags, category=category, scope=scope)

    def forget(self, memory_id: int) -> bool:
        return self.store.delete_memory(memory_id)


def build_memory_provider(name: str, database: Path) -> MemoryProvider:
    normalized = name.strip().lower()
    if normalized == "sqlite":
        return SQLiteMemoryProvider(database)
    raise ValueError(f"unsupported memory provider: {name}")


class MemoryManager:
    def __init__(self, database: Path) -> None:
        self.store = SessionStore(database)
        self.provider: MemoryProvider = SQLiteMemoryProvider(database)

    def remember(
        self, content: str, *, category: str = "general", tags: str = "", scope: str = "global"
    ) -> int:
        return self.provider.remember(content, category=category, tags=tags, scope=scope)

    def recall(
        self, query: str, *, limit: int = 5, scope: str | None = None
    ) -> list[dict[str, Any]]:
        return self.provider.recall(query, limit=limit, scope=scope)

    def forget(self, memory_id: int) -> bool:
        return self.provider.forget(memory_id)

    def update(
        self,
        memory_id: int,
        content: str,
        *,
        category: str | None = None,
        tags: str | None = None,
    ) -> bool:
        return self.store.update_memory(memory_id, content, category=category, tags=tags)

    def curated_context(self, *, scope: str = "global", max_chars: int = 1800) -> list[str]:
        if not 100 <= max_chars <= 6000:
            raise ValueError("max_chars must be between 100 and 6000")
        rows = self.store.list_memories(limit=200)
        selected = [
            row
            for row in rows
            if row.get("memory_type") in {"profile", "preference", "project_fact"}
            and row.get("scope", "global") in {"global", scope}
            and float(row.get("confidence", 0.7)) >= 0.6
        ]
        result: list[str] = []
        used = 0
        for row in selected:
            line = f"[{row['memory_type']}] {str(row['content']).strip()}"
            if not line or used + len(line) > max_chars:
                continue
            result.append(line)
            used += len(line)
        return result


class PromptMemory:
    def __init__(
        self, database: Path, lessons: Any, *, max_chars: int = 1800, scope: str = "global"
    ) -> None:
        self.database = database
        self.lessons = lessons
        self.max_chars = max_chars
        self.scope = scope

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
            rows = manager.store.search_memories_any(terms, 8, scope=self.scope)
            items.extend(
                f"[{row.get('memory_type', 'fact')}] {row['content']}"
                for row in rows
                if row.get("category") != "lesson" and float(row.get("confidence", 0.7)) >= 0.55
            )
            items.extend(manager.curated_context(scope=self.scope, max_chars=self.max_chars))
        result: list[str] = []
        used = 0
        for item in items:
            clean = str(item).strip()
            if not clean or used + len(clean) > self.max_chars:
                continue
            result.append(clean)
            used += len(clean)
        return list(dict.fromkeys(result))

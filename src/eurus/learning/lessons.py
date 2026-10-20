from __future__ import annotations

import re
from pathlib import Path

from eurus.memory.manager import MemoryManager

CATEGORY = "lesson"
MAX_QUERY_TERMS = 12
STOPWORDS = frozenset(
    [
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "your",
        "have",
        "will",
        "can",
        "not",
        "are",
        "was",
        "you",
        "any",
        "all",
        "yang",
        "dan",
        "untuk",
        "dengan",
        "dari",
        "pada",
        "ini",
        "itu",
        "atau",
        "ada",
        "tidak",
        "akan",
        "bisa",
        "saya",
        "kamu",
        "buat",
    ]
)


def _terms(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for word in re.findall(r"\w+", text.lower()):
        if len(word) >= 3 and word not in STOPWORDS and word.isalnum():
            seen.setdefault(word)
    return list(seen)


def _similarity(left: str, right: str) -> float:
    a, b = set(re.findall(r"\w+", left.lower())), set(re.findall(r"\w+", right.lower()))
    return len(a & b) / len(a | b) if a and b else 0.0


class LessonStore:
    def __init__(self, database: Path, *, limit: int = 5, max_lessons: int = 500) -> None:
        self.database = database
        self.limit = limit
        self.max_lessons = max_lessons
        self._memory: MemoryManager | None = None

    def _manager(self) -> MemoryManager:
        if self._memory is None:
            self._memory = MemoryManager(self.database)
        return self._memory

    def relevant(self, task: str) -> list[str]:
        terms = _terms(task)[:MAX_QUERY_TERMS]
        if self.limit < 1 or not terms or not self.database.exists():
            return []
        rows = self._manager().store.search_memories_any(terms, self.limit, category=CATEGORY)
        return [row["content"] for row in rows]

    def all(self) -> list[str]:
        if not self.database.exists():
            return []
        rows = self._manager().store.list_memories(category=CATEGORY, limit=self.max_lessons)
        return [row["content"] for row in rows]

    def is_duplicate(self, lesson: str) -> bool:
        return any(_similarity(lesson, known) >= 0.7 for known in self.all())

    def add(self, lesson: str) -> int | None:
        if self.is_duplicate(lesson) or len(self.all()) >= self.max_lessons:
            return None
        return self._manager().store.add_memory(
            lesson,
            tags="auto",
            category=CATEGORY,
            memory_type="lesson",
            source="reflector",
            confidence=0.75,
        )

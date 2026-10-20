from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class ConcurrencyLimiter:
    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("concurrency limit must be at least 1")
        self.limit = limit
        self._semaphore = asyncio.Semaphore(limit)
        self.active = 0
        self.peak_active = 0

    async def run(self, operation: Callable[[], Awaitable[T]]) -> T:
        async with self._semaphore:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            try:
                return await operation()
            finally:
                self.active -= 1

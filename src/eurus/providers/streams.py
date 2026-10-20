from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import TypeVar

T = TypeVar("T")


@asynccontextmanager
async def closing_stream(source: AsyncIterator[T]) -> AsyncIterator[AsyncIterator[T]]:
    try:
        yield source
    finally:
        if isinstance(source, AsyncGenerator):
            await source.aclose()

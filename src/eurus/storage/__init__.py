from typing import Any

__all__ = ["ConcurrentRunUpdate", "Database", "EventStore", "RunStore"]


def __getattr__(name: str) -> Any:
    if name == "Database":
        from eurus.storage.database import Database

        return Database
    if name == "EventStore":
        from eurus.storage.events import EventStore

        return EventStore
    if name in {"ConcurrentRunUpdate", "RunStore"}:
        from eurus.storage.runs import ConcurrentRunUpdate, RunStore

        return {"ConcurrentRunUpdate": ConcurrentRunUpdate, "RunStore": RunStore}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TypeAlias

from eurus.storage.migrations import apply_migrations

DatabasePath: TypeAlias = str | Path


class Database:
    def __init__(self, path: DatabasePath) -> None:
        self.path = str(path)
        if self.path == ":memory:":
            raise ValueError("durable runtime storage requires a file-backed SQLite database")
        target = Path(self.path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        target.touch(mode=0o600, exist_ok=True)
        self.path = str(target)
        with suppress(OSError):
            os.chmod(target, 0o600)
        with self.connection() as connection:
            apply_migrations(connection)

    def _new_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def connect(self) -> sqlite3.Connection:
        return self._new_connection()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

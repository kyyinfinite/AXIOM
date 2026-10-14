from __future__ import annotations

import json
import os
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class AuditJournal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch(mode=0o600)
        with suppress(OSError):
            os.chmod(self.path, 0o600)

    def record(self, event: str, **fields: Any) -> None:
        item = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

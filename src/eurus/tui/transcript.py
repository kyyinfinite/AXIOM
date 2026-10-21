from __future__ import annotations

import re
from collections.abc import Callable

MAX_LINES = 6000
KEEP_LINES = 5000
OSC = re.compile(r"\x1b\].*?(?:\x07|\x1b\\)")
PRIVATE_MODE = re.compile(r"\x1b\[\?[0-9;]*[hl]")


class TranscriptSink:
    encoding = "utf-8"
    errors = "replace"

    def __init__(self) -> None:
        self.lines: list[str] = [""]
        self.version = 0
        self.on_change: Callable[[], None] | None = None

    def write(self, text: str) -> int:
        clean = PRIVATE_MODE.sub("", OSC.sub("", text)).replace("\r", "")
        parts = clean.split("\n")
        self.lines[-1] += parts[0]
        self.lines.extend(parts[1:])
        if len(self.lines) > MAX_LINES:
            del self.lines[:-KEEP_LINES]
        self._changed()
        return len(text)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False

    def clear(self) -> None:
        self.lines = [""]
        self._changed()

    def _changed(self) -> None:
        self.version += 1
        if self.on_change is not None:
            self.on_change()

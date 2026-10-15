from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

ColorLevel = Literal["truecolor", "256", "basic", "none"]

COMPACT_WIDTH = 60
NARROW_WIDTH = 45


@dataclass(frozen=True, slots=True)
class TerminalCapabilities:
    color: ColorLevel
    unicode: bool
    width: int

    @property
    def compact(self) -> bool:
        return self.width < COMPACT_WIDTH

    @property
    def narrow(self) -> bool:
        return self.width < NARROW_WIDTH


def detect_color(environ: Mapping[str, str], is_tty: bool) -> ColorLevel:
    if "NO_COLOR" in environ or not is_tty:
        return "none"
    term = environ.get("TERM", "").lower()
    if term in ("", "dumb"):
        return "none"
    if environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return "truecolor"
    if "256color" in term:
        return "256"
    return "basic"


def detect_unicode(environ: Mapping[str, str], encoding: str) -> bool:
    if encoding.lower().replace("-", "").startswith("utf"):
        return True
    locale = (environ.get("LC_ALL") or environ.get("LC_CTYPE") or environ.get("LANG") or "").lower()
    return "utf-8" in locale or "utf8" in locale


def detect(
    environ: Mapping[str, str] | None = None,
    *,
    width: int | None = None,
    encoding: str | None = None,
    is_tty: bool | None = None,
) -> TerminalCapabilities:
    env = os.environ if environ is None else environ
    tty = sys.stdout.isatty() if is_tty is None else is_tty
    enc = (getattr(sys.stdout, "encoding", "") or "") if encoding is None else encoding
    columns = shutil.get_terminal_size((80, 24)).columns if width is None else width
    return TerminalCapabilities(
        color=detect_color(env, tty),
        unicode=detect_unicode(env, enc),
        width=max(columns, 20),
    )

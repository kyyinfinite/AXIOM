from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Segment:
    text: str
    flexible: bool = False


def clip(text: str, limit: int, ellipsis: str = "...") -> str:
    flat = " ".join(text.split())
    if limit <= 0:
        return ""
    if len(flat) <= limit:
        return flat
    keep = limit - len(ellipsis)
    if keep <= 0:
        return flat[:limit]
    return flat[:keep] + ellipsis


def shorten_path(path: Path | str, limit: int, ellipsis: str = "...") -> str:
    text = str(path)
    home = str(Path.home())
    offset = len(home)
    if text == home or text.startswith(home + os.sep):
        text = "~" + text[offset:]
    if len(text) <= limit:
        return text
    keep = max(limit - len(ellipsis), 1)
    return ellipsis + text[-keep:]


def fit_segments(segments: Sequence[Segment], width: int, separator: str = " | ") -> list[str]:
    chosen: list[str] = []
    used = 0
    for segment in segments:
        if not chosen:
            first = clip(segment.text, width)
            chosen.append(first)
            used = len(first)
            continue
        extra = len(separator) + len(segment.text)
        if used + extra <= width:
            chosen.append(segment.text)
            used += extra
        elif segment.flexible and width - used - len(separator) >= 8:
            room = width - used - len(separator)
            chosen.append(clip(segment.text, room))
            used = width
    return chosen


def format_tokens(count: int) -> str:
    if count < 1000:
        return str(count)
    if count < 100_000:
        return f"{count / 1000:.1f}k"
    return f"{count // 1000}k"

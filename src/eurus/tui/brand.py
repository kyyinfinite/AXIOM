from __future__ import annotations

import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from eurus.tui.theme import Theme

Rgb = tuple[int, int, int]
Cell = tuple[str, Rgb | None]
Grid = list[list[Cell]]

GRADIENT: tuple[Rgb, ...] = ((208, 138, 106), (230, 223, 210), (79, 179, 191))
ART_PATH = Path(__file__).with_name("banner.ans")
NAME = "EURUS"
TAGLINE = "AUTONOMOUS AI AGENT"
GAP = 3
MIN_ART_ROWS = 12
MAX_ART_ROWS = 24
CHROME_ROWS = 12
SGR = re.compile(r"\x1b\[([0-9;]*)m")
BLANK: Cell = (" ", None)


def spaced(text: str) -> str:
    return "   ".join(" ".join(word) for word in text.split())


TAGLINE_SPACED = spaced(TAGLINE)
TEXT_WIDTH = len(TAGLINE_SPACED)


def _sample(stops: Sequence[Rgb], position: float) -> Rgb:
    if len(stops) == 1:
        return stops[0]
    scaled = min(max(position, 0.0), 1.0) * (len(stops) - 1)
    index = min(int(scaled), len(stops) - 2)
    fraction = scaled - index
    low, high = stops[index], stops[index + 1]
    return (
        round(low[0] + (high[0] - low[0]) * fraction),
        round(low[1] + (high[1] - low[1]) * fraction),
        round(low[2] + (high[2] - low[2]) * fraction),
    )


def gradient(text: str, stops: Sequence[Rgb] = GRADIENT) -> Text:
    result = Text()
    last = max(len(text) - 1, 1)
    for index, char in enumerate(text):
        red, green, blue = _sample(stops, index / last)
        result.append(char, style=f"bold #{red:02x}{green:02x}{blue:02x}")
    return result


def parse_ansi(source: str) -> Grid:
    grid: Grid = []
    for raw in source.split("\n"):
        row: list[Cell] = []
        color: Rgb | None = None
        position = 0
        for match in SGR.finditer(raw):
            row.extend((char, color) for char in raw[position : match.start()])
            position = match.end()
            params = [int(part) if part else 0 for part in match.group(1).split(";")]
            index = 0
            while index < len(params):
                code = params[index]
                if code == 0:
                    color = None
                    index += 1
                elif code == 38 and index + 4 < len(params) and params[index + 1] == 2:
                    color = (params[index + 2], params[index + 3], params[index + 4])
                    index += 5
                else:
                    index += 1
        row.extend((char, color) for char in raw[position:])
        grid.append(row)
    return grid


def _blank(row: list[Cell]) -> bool:
    return all(char == " " for char, _ in row)


def trim(grid: Grid) -> Grid:
    rows = list(grid)
    while rows and _blank(rows[0]):
        rows.pop(0)
    while rows and _blank(rows[-1]):
        rows.pop()
    if not rows:
        return []
    filled = [row for row in rows if not _blank(row)]
    left = min(next(i for i, (char, _) in enumerate(row) if char != " ") for row in filled)
    right = max(max(i for i, (char, _) in enumerate(row) if char != " ") for row in filled) + 1
    width = right - left
    trimmed: Grid = []
    for row in rows:
        part = row[left:right]
        trimmed.append(part + [BLANK] * (width - len(part)))
    return trimmed


def scale(grid: Grid, rows: int) -> Grid:
    height = len(grid)
    if not grid or rows >= height:
        return grid
    width = len(grid[0])
    cols = max(round(width * rows / height), 1)
    return [
        [
            grid[min(int((r + 0.5) * height / rows), height - 1)][
                min(int((c + 0.5) * width / cols), width - 1)
            ]
            for c in range(cols)
        ]
        for r in range(rows)
    ]


@lru_cache(maxsize=4)
def load_art(path: Path = ART_PATH) -> Grid:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return []
    return trim(parse_ansi(source.rstrip("\n")))


def art_text(grid: Grid) -> Text:
    lines: list[Text] = []
    for row in grid:
        line = Text()
        for char, color in row:
            if color is None or char == " ":
                line.append(char)
            else:
                line.append(char, style=f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}")
        lines.append(line)
    return Text("\n").join(lines)


def place(art: Grid, width: int, height: int) -> Grid | None:
    if not art:
        return None
    room = height if height > 0 else 999
    limit = min(MAX_ART_ROWS, room - CHROME_ROWS, len(art))
    for rows in range(limit, MIN_ART_ROWS - 1, -1):
        scaled = scale(art, rows)
        if len(scaled[0]) + GAP + TEXT_WIDTH <= width:
            return scaled
    return None


def wordmark(version: str, theme: Theme) -> Text:
    name = gradient(NAME)
    if version:
        name.append(f"  v{version}", style=theme.amber)
    return name


def tagline(width: int, theme: Theme) -> Text:
    text = TAGLINE_SPACED if width >= TEXT_WIDTH else TAGLINE
    return Text(text, style=theme.dim)


def flow(width: int, theme: Theme) -> Text:
    return gradient(theme.glyphs.rule * max(min(width, TEXT_WIDTH), 1))


def header(
    version: str, width: int, height: int, theme: Theme, color: bool = True
) -> RenderableType:
    block = Group(wordmark(version, theme), tagline(width, theme), flow(width, theme))
    placed = place(load_art(), width, height) if color else None
    if placed is None:
        return block
    grid = Table.grid(padding=(0, GAP))
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True, vertical="middle")
    grid.add_row(art_text(placed), block)
    return grid

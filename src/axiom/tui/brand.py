from __future__ import annotations

from collections.abc import Sequence

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from axiom.tui.theme import Theme

Rgb = tuple[int, int, int]

GRADIENT: tuple[Rgb, ...] = ((34, 211, 238), (96, 165, 250), (167, 139, 250))
MARK: tuple[str, str] = (" /\\  \\ /", "/--\\ / \\")
BANNER: tuple[str, ...] = (
    "▄▄▄      ▒██   ██▒  ▄████ ▓█████  ███▄    █",
    "▒████▄    ▒▒ █ █ ▒░ ██▒ ▀█▒▓█   ▀  ██ ▀█   █",
    "▒██  ▀█▄  ░░  █   ░▒██░▄▄▄░▒███   ▓██  ▀█ ██▒",
    "░██▄▄▄▄██  ░ █ █ ▒ ░▓█  ██▓▒▓█  ▄ ▓██▒  ▐▌██▒",
    " ▓█   ▓██▒▒██▒ ▒██▒░▒▓███▀▒░▒████▒▒██░   ▓██░",
    " ▒▒   ▓▒█░▒▒ ░ ░▓ ░ ░▒   ▒ ░░ ▒░ ░░ ▒░   ▒ ▒",
    "  ▒   ▒▒ ░░░   ░▒ ░  ░   ░  ░ ░  ░░ ░░   ░ ▒░",
    "  ░   ▒    ░    ░  ░ ░   ░    ░      ░   ░ ░",
    "      ░  ░ ░    ░        ░    ░  ░         ░",
)
BANNER_GRADIENT: tuple[Rgb, ...] = (
    (34, 211, 238),
    (66, 133, 244),
    (155, 114, 203),
    (217, 101, 144),
    (251, 146, 60),
)
BANNER_WIDTH = max(len(row) for row in BANNER)
BANNER_MIN_WIDTH = BANNER_WIDTH + 1
COMPACT_COLUMNS = 19
COMPACT_MIN_WIDTH = 38
NAME = "AXagent"
SUBTITLE = "Autonomous Developer Agent"
SUBTITLE_MIN_WIDTH = 38


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


def render_banner(rows_text: Sequence[str]) -> Text:
    rows = len(rows_text)
    width = max(len(row) for row in rows_text)
    span = width + rows * 2
    lines: list[Text] = []
    for row_index, row in enumerate(rows_text):
        line = Text()
        for column, char in enumerate(row):
            if char == " ":
                line.append(char)
                continue
            red, green, blue = _sample(BANNER_GRADIENT, (column + row_index * 2) / span)
            line.append(char, style=f"#{red:02x}{green:02x}{blue:02x}")
        lines.append(line)
    return Text("\n").join(lines)


def banner() -> Text:
    return render_banner(BANNER)


def compact_banner(version: str, theme: Theme) -> RenderableType:
    art = render_banner(tuple(row[:COMPACT_COLUMNS] for row in BANNER))
    side = Group(
        wordmark(version, theme),
        Text("Autonomous", style=theme.dim),
        Text("Developer Agent", style=theme.dim),
    )
    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True, vertical="middle")
    grid.add_row(art, side)
    return grid


def mark() -> Text:
    return Text("\n").join(gradient(row) for row in MARK)


def wordmark(version: str, theme: Theme) -> Text:
    name = gradient("AX")
    name.append("agent", style=f"bold {theme.text}")
    if version:
        name.append(f"  v{version}", style=theme.orange)
    return name


def header(version: str, width: int, theme: Theme, unicode: bool = True) -> RenderableType:
    if unicode and width >= BANNER_MIN_WIDTH:
        parts: list[RenderableType] = [banner(), Text(), wordmark(version, theme)]
        if width >= SUBTITLE_MIN_WIDTH:
            parts.append(Text(SUBTITLE, style=theme.dim))
        return Group(*parts)
    if unicode and width >= COMPACT_MIN_WIDTH:
        return compact_banner(version, theme)
    lines: list[RenderableType] = [wordmark(version, theme)]
    if width >= SUBTITLE_MIN_WIDTH:
        lines.append(Text(SUBTITLE, style=theme.dim))
    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True)
    grid.add_column()
    grid.add_row(mark(), Group(*lines))
    return grid

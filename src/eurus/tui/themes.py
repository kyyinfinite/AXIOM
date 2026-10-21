from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any

from eurus import paths
from eurus.tui.theme import (
    ASCII_GLYPHS,
    GLYPH_KEYS,
    GLYPH_SETS,
    UNICODE_GLYPHS,
    Glyphs,
    Theme,
)

NON_COLOR = {"glyphs", "code_theme"}
PALETTE_KEYS = tuple(item.name for item in fields(Theme) if item.name not in NON_COLOR)
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
NAME = re.compile(r"^[a-z0-9_-]{1,40}$")
CONTROL = re.compile(r"[\x00-\x1f\x7f]")
DEFAULT = "eurus"
STATE_FILE = "theme"
GLYPH_FILE = "glyphs"
RESERVED = frozenset({"preview", "new", "path", "termux", "reset", "list"})
MAX_DEPTH = 8
MAX_GLYPH = 12

BUILTIN: dict[str, dict[str, str]] = {
    "eurus": {},
    "midnight": {
        "teal": "#7fc4c4",
        "steel": "#b9b4c9",
        "copper": "#b4a0e0",
        "amber": "#e0b070",
        "green": "#8bc98f",
        "yellow": "#e3c26b",
        "red": "#e07a7a",
        "text": "#d7dce5",
        "dim": "#8a93a3",
        "border": "#363e4d",
        "surface": "#1b212c",
        "background": "#0e1117",
    },
    "paper": {
        "teal": "#2e7f88",
        "steel": "#6b5f4a",
        "copper": "#a8552f",
        "amber": "#a8731a",
        "green": "#3d7a46",
        "yellow": "#8a6d12",
        "red": "#b03a3a",
        "text": "#2b2622",
        "dim": "#7a7268",
        "border": "#cfc6b6",
        "surface": "#e9e2d3",
        "background": "#f6f1e7",
    },
    "mono": {
        "teal": "#cccccc",
        "steel": "#bbbbbb",
        "copper": "#ffffff",
        "amber": "#ffffff",
        "green": "#dddddd",
        "yellow": "#eeeeee",
        "red": "#ffffff",
        "text": "#ffffff",
        "dim": "#999999",
        "border": "#666666",
        "surface": "#1a1a1a",
        "background": "#000000",
    },
    "nord": {
        "teal": "#88c0d0",
        "steel": "#d8dee9",
        "copper": "#81a1c1",
        "amber": "#d08770",
        "green": "#a3be8c",
        "yellow": "#ebcb8b",
        "red": "#bf616a",
        "text": "#d8dee9",
        "dim": "#7b88a1",
        "border": "#434c5e",
        "surface": "#3b4252",
        "background": "#2e3440",
    },
    "dracula": {
        "teal": "#8be9fd",
        "steel": "#f8f8f2",
        "copper": "#bd93f9",
        "amber": "#ffb86c",
        "green": "#50fa7b",
        "yellow": "#f1fa8c",
        "red": "#ff5555",
        "text": "#f8f8f2",
        "dim": "#6272a4",
        "border": "#44475a",
        "surface": "#343746",
        "background": "#282a36",
    },
    "gruvbox": {
        "teal": "#8ec07c",
        "steel": "#d5c4a1",
        "copper": "#fe8019",
        "amber": "#fabd2f",
        "green": "#b8bb26",
        "yellow": "#fabd2f",
        "red": "#fb4934",
        "text": "#ebdbb2",
        "dim": "#928374",
        "border": "#504945",
        "surface": "#3c3836",
        "background": "#282828",
    },
    "tokyo-night": {
        "teal": "#7dcfff",
        "steel": "#a9b1d6",
        "copper": "#bb9af7",
        "amber": "#ff9e64",
        "green": "#9ece6a",
        "yellow": "#e0af68",
        "red": "#f7768e",
        "text": "#c0caf5",
        "dim": "#565f89",
        "border": "#3b4261",
        "surface": "#24283b",
        "background": "#1a1b26",
    },
    "catppuccin": {
        "teal": "#94e2d5",
        "steel": "#bac2de",
        "copper": "#cba6f7",
        "amber": "#fab387",
        "green": "#a6e3a1",
        "yellow": "#f9e2af",
        "red": "#f38ba8",
        "text": "#cdd6f4",
        "dim": "#7f849c",
        "border": "#45475a",
        "surface": "#313244",
        "background": "#1e1e2e",
    },
}

BUILTIN_CODE: dict[str, str] = {
    "midnight": "one-dark",
    "paper": "friendly",
    "mono": "bw",
    "nord": "nord",
    "dracula": "dracula",
    "gruvbox": "gruvbox-dark",
    "tokyo-night": "one-dark",
    "catppuccin": "material",
}

DESCRIPTIONS: dict[str, str] = {
    "eurus": "Warm copper and teal, the default",
    "midnight": "Cool violet on deep blue",
    "paper": "Light theme on warm paper",
    "mono": "Pure grayscale",
    "nord": "Arctic blues",
    "dracula": "Vivid purple and green",
    "gruvbox": "Retro warm contrast",
    "tokyo-night": "Neon city blues",
    "catppuccin": "Soft pastel mocha",
}


@dataclass(frozen=True, slots=True)
class ThemeSpec:
    description: str = ""
    extends: str = ""
    colors: dict[str, str] = field(default_factory=dict)
    code_theme: str = ""
    glyph_set: str = ""
    glyphs: dict[str, str] = field(default_factory=dict)
    font: str = ""


def user_dir(home: Path | None = None) -> Path:
    return (home or paths.home_dir()) / "themes"


def available(home: Path | None = None) -> list[str]:
    found = {
        item.stem.lower()
        for item in user_dir(home).glob("*.toml")
        if NAME.match(item.stem.lower()) and item.stem.lower() not in RESERVED
    }
    return [*BUILTIN, *sorted(found - set(BUILTIN))]


def describe_theme(name: str, home: Path | None = None) -> str:
    if name in DESCRIPTIONS:
        return DESCRIPTIONS[name]
    try:
        spec = read_spec(user_dir(home) / f"{name}.toml")
    except ValueError:
        return "Unreadable theme file"
    return spec.description or "Custom theme"


@lru_cache(maxsize=1)
def code_styles() -> frozenset[str]:
    return frozenset(import_module("pygments.styles").get_all_styles())


def _text(value: Any, limit: int = 80) -> str:
    if not isinstance(value, str):
        return ""
    return CONTROL.sub("", value).strip()[:limit]


def _section(data: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def read_spec(path: Path) -> ThemeSpec:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Cannot read theme {path.name}: {exc}") from exc
    meta = _section(data, "meta")
    style = _section(data, "style")
    table = data.get("colors")
    if not isinstance(table, dict):
        table = {key: value for key, value in data.items() if not isinstance(value, dict)}
    colors = {
        key: value.lower()
        for key, value in table.items()
        if key in PALETTE_KEYS and isinstance(value, str) and HEX.match(value)
    }
    overrides = style.get("glyph")
    glyphs = (
        {
            key: value
            for key, value in overrides.items()
            if key in GLYPH_KEYS
            and isinstance(value, str)
            and 0 < len(value) <= MAX_GLYPH
            and not CONTROL.search(value)
        }
        if isinstance(overrides, dict)
        else {}
    )
    font = _section(data, "font")
    return ThemeSpec(
        description=_text(meta.get("description"), 60),
        extends=_text(meta.get("extends"), 40).lower(),
        colors=colors,
        code_theme=_text(style.get("code"), 40),
        glyph_set=_text(style.get("glyphs"), 10).lower(),
        glyphs=glyphs,
        font=_text(font.get("name"), 40).lower(),
    )


def read_colors(path: Path) -> dict[str, str]:
    return read_spec(path).colors


def _apply(theme: Theme, changes: Mapping[str, Any]) -> Theme:
    return replace(theme, **changes)


def _glyph_base(choice: str, unicode: bool) -> Glyphs:
    if not unicode or choice == "ascii":
        return ASCII_GLYPHS
    return GLYPH_SETS.get(choice, UNICODE_GLYPHS)


def _build(
    key: str, unicode: bool, glyph_set: str, home: Path | None, trail: tuple[str, ...]
) -> tuple[Theme, ThemeSpec | None]:
    if len(trail) >= MAX_DEPTH or key in trail:
        raise ValueError(f"Theme inheritance loop: {' -> '.join((*trail, key))}")
    if key in BUILTIN:
        changes: dict[str, Any] = dict(BUILTIN[key])
        if key in BUILTIN_CODE:
            changes["code_theme"] = BUILTIN_CODE[key]
        base = Theme(glyphs=_glyph_base(glyph_set, unicode))
        return _apply(base, changes), None
    path = user_dir(home) / f"{key}.toml"
    if not NAME.match(key) or key in RESERVED or not path.is_file():
        raise ValueError(f"Unknown theme: {key}")
    spec = read_spec(path)
    parent = spec.extends or DEFAULT
    theme, _ = _build(parent, unicode, glyph_set or spec.glyph_set, home, (*trail, key))
    changes = dict(spec.colors)
    if spec.code_theme in code_styles():
        changes["code_theme"] = spec.code_theme
    chosen = glyph_set or spec.glyph_set
    if chosen in GLYPH_SETS:
        changes["glyphs"] = _glyph_base(chosen, unicode)
    theme = _apply(theme, changes)
    extra: dict[str, Any] = {
        name: value for name, value in spec.glyphs.items() if unicode or value.isascii()
    }
    if extra:
        theme = _apply(theme, {"glyphs": replace(theme.glyphs, **extra)})
    return theme, spec


def resolve(
    name: str, unicode: bool = True, home: Path | None = None, glyph_set: str = ""
) -> Theme:
    key = name.strip().lower() or DEFAULT
    return _build(key, unicode, glyph_set, home, ())[0]


def font_for(name: str, home: Path | None = None) -> str:
    key = name.strip().lower() or DEFAULT
    path = user_dir(home) / f"{key}.toml"
    if key in BUILTIN or not NAME.match(key) or not path.is_file():
        return ""
    try:
        return read_spec(path).font
    except ValueError:
        return ""


def active_name(environ: Mapping[str, str] | None = None, home: Path | None = None) -> str:
    env = os.environ if environ is None else environ
    chosen = env.get("EURUS_THEME", "").strip()
    if chosen:
        return chosen
    base = home or paths.home_dir(environ)
    try:
        stored = (base / STATE_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        stored = ""
    return stored or DEFAULT


def save_active(name: str, home: Path | None = None) -> None:
    _write_state(STATE_FILE, name, home)


def glyph_choice(environ: Mapping[str, str] | None = None, home: Path | None = None) -> str:
    env = os.environ if environ is None else environ
    chosen = env.get("EURUS_GLYPHS", "").strip().lower()
    if chosen in GLYPH_SETS:
        return chosen
    base = home or paths.home_dir(environ)
    try:
        stored = (base / GLYPH_FILE).read_text(encoding="utf-8").strip().lower()
    except OSError:
        return ""
    return stored if stored in GLYPH_SETS else ""


def save_glyphs(choice: str, home: Path | None = None) -> None:
    base = home or paths.home_dir()
    if choice in GLYPH_SETS:
        _write_state(GLYPH_FILE, choice, base)
        return
    try:
        (base / GLYPH_FILE).unlink(missing_ok=True)
    except OSError:
        return


def _write_state(filename: str, value: str, home: Path | None) -> None:
    base = home or paths.home_dir()
    try:
        base.mkdir(parents=True, exist_ok=True)
        (base / filename).write_text(value + "\n", encoding="utf-8")
    except OSError:
        return


def load_active(
    unicode: bool = True,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Theme:
    base = home or paths.home_dir(environ)
    choice = glyph_choice(environ, base)
    try:
        return resolve(active_name(environ, base), unicode, base, choice)
    except ValueError:
        return Theme(glyphs=_glyph_base(choice, unicode))


TEMPLATE = """\
[meta]
description = "My {name} theme"
extends = "{base}"

[colors]
# Any key can be removed; missing keys come from "extends".
copper = "{copper}"
teal = "{teal}"
text = "{text}"
dim = "{dim}"
border = "{border}"
surface = "{surface}"
background = "{background}"
green = "{green}"
yellow = "{yellow}"
red = "{red}"

[style]
# Pygments style for code blocks, for example: dracula, nord, one-dark, monokai
code = "{code}"
# unicode, ascii or nerd (nerd needs a Nerd Font, see /font)
glyphs = "unicode"

[style.glyph]
# prompt = "❯"

[font]
# Catalog key from /font list; only a suggestion, nothing is downloaded automatically
# name = "jetbrains-mono-nerd"
"""


def scaffold(name: str, base: str = DEFAULT, home: Path | None = None) -> Path:
    key = name.strip().lower()
    if not NAME.match(key) or key in RESERVED or key in BUILTIN:
        raise ValueError(f"Invalid theme name: {name!r}")
    target = user_dir(home) / f"{key}.toml"
    if target.exists():
        raise ValueError(f"Theme already exists: {target}")
    parent = resolve(base, True, home)
    folder = user_dir(home)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        target.write_text(
            TEMPLATE.format(
                name=key,
                base=base.strip().lower() or DEFAULT,
                code=parent.code_theme,
                **{item: getattr(parent, item) for item in PALETTE_KEYS},
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        raise ValueError(f"Cannot write theme: {exc}") from exc
    return target


TERMUX_ANSI = (
    "background",
    "red",
    "green",
    "yellow",
    "teal",
    "copper",
    "teal",
    "text",
    "dim",
    "red",
    "green",
    "yellow",
    "teal",
    "copper",
    "teal",
    "text",
)


def termux_colors(theme: Theme) -> str:
    lines = [
        f"background={theme.background}",
        f"foreground={theme.text}",
        f"cursor={theme.copper}",
    ]
    lines.extend(
        f"color{index}={getattr(theme, key) if key != 'background' else theme.surface}"
        for index, key in enumerate(TERMUX_ANSI)
    )
    return "\n".join(lines) + "\n"


def write_termux_colors(theme: Theme, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_name(path.name + ".eurus-backup")
    if path.exists() and not backup.exists():
        backup.write_bytes(path.read_bytes())
    path.write_text(termux_colors(theme), encoding="utf-8")

import asyncio
import io
import re

import pytest
from rich.console import Console

from eurus.config import load_config
from eurus.tui.brand import hex_rgb, load_art, tint
from eurus.tui.renderer import TerminalRenderer
from eurus.tui.session import InteractiveSession
from eurus.tui.terminal import (
    RESET_COLORS,
    Recolor,
    color_sequence,
    recolor_mode,
    supported,
    title_sequence,
)
from eurus.tui.theme import Theme
from eurus.tui.themes import (
    BUILTIN,
    PALETTE_KEYS,
    active_name,
    available,
    load_active,
    resolve,
    save_active,
)

HEX = re.compile(r"^#[0-9a-f]{6}$")


class Tty(io.StringIO):
    def isatty(self):
        return True


def test_builtin_themes_are_complete_valid_hex():
    for name in BUILTIN:
        theme = resolve(name)
        for key in PALETTE_KEYS:
            assert HEX.match(getattr(theme, key).lower()), (name, key)
    assert resolve("paper").background != resolve("eurus").background
    assert resolve("eurus") == Theme()


def test_unknown_and_traversal_names_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        resolve("nope", home=tmp_path)
    with pytest.raises(ValueError):
        resolve("../eurus", home=tmp_path)


def test_user_theme_overrides_valid_colors_only(tmp_path):
    folder = tmp_path / "themes"
    folder.mkdir()
    (folder / "ember.toml").write_text(
        '[colors]\ncopper = "#ff0000"\nred = "red"\nbogus = "#ffffff"\n', encoding="utf-8"
    )
    theme = resolve("ember", home=tmp_path)
    assert theme.copper == "#ff0000" and theme.red == Theme().red
    assert "ember" in available(tmp_path) and available(tmp_path)[0] == "eurus"
    (folder / "broken.toml").write_text("= nope", encoding="utf-8")
    with pytest.raises(ValueError):
        resolve("broken", home=tmp_path)


def test_active_theme_precedence_and_persistence(tmp_path):
    assert active_name({}, tmp_path) == "eurus"
    save_active("midnight", tmp_path)
    assert active_name({}, tmp_path) == "midnight"
    assert active_name({"EURUS_THEME": "paper"}, tmp_path) == "paper"
    assert load_active(True, {}, tmp_path) == resolve("midnight")
    assert load_active(True, {"EURUS_THEME": "missing"}, tmp_path) == Theme()


def test_art_tint_stays_between_background_and_text():
    art = load_art()
    for name in ("eurus", "paper"):
        theme = resolve(name)
        low, high = hex_rgb(theme.background), hex_rgb(theme.text)
        for row in tint(art, theme):
            for _, color in row:
                if color is not None:
                    for channel, a, b in zip(color, low, high, strict=True):
                        assert min(a, b) <= channel <= max(a, b)


def test_recolor_modes_and_support():
    assert recolor_mode({}) == "auto" and recolor_mode({"EURUS_RECOLOR": "NEVER"}) == "never"
    assert recolor_mode({"EURUS_RECOLOR": "weird"}) == "auto"
    assert not supported(io.StringIO(), "always")
    assert supported(Tty(), "always", {})
    assert not supported(Tty(), "never", {})
    assert supported(Tty(), "auto", {"TERM": "xterm-256color"})
    assert not supported(Tty(), "auto", {"TERM": "screen-256color"})
    assert not supported(Tty(), "auto", {"TERM": "xterm", "NO_COLOR": "1"})
    assert not supported(Tty(), "auto", {"TERM": "xterm", "TMUX": "/tmp/x"})


def test_recolor_applies_and_restores_once():
    stream = Tty()
    recolor = Recolor(stream, "always", {})
    theme = Theme()
    recolor.apply(theme, "Eurus - demo")
    out = stream.getvalue()
    assert color_sequence(theme) in out and "Eurus - demo" in out
    assert recolor.active
    before = len(stream.getvalue())
    recolor.restore()
    assert stream.getvalue().count(RESET_COLORS) == 1 and not recolor.active
    recolor.restore()
    assert stream.getvalue().count(RESET_COLORS) == 1 and len(stream.getvalue()) > before


def test_recolor_is_silent_when_disabled():
    stream = io.StringIO()
    recolor = Recolor(stream, "auto", {})
    recolor.apply(Theme(), "x")
    recolor.restore()
    assert stream.getvalue() == ""


def test_title_sequence_strips_control_characters():
    assert "\x1b]2;ab" in title_sequence("a\x1b\x07b") and "\x1b]2;a\x1b" not in title_sequence(
        "a\x1b\x07b"
    )


def build_session(tmp_path, monkeypatch):
    monkeypatch.setenv("EURUS_HOME", str(tmp_path))
    monkeypatch.delenv("EURUS_THEME", raising=False)
    config = load_config(cwd=tmp_path, environ={"EURUS_HOME": str(tmp_path)})
    console = Console(record=True, width=60, force_terminal=False)
    return InteractiveSession(config, TerminalRenderer(console)), console


def test_theme_command_lists_switches_and_persists(tmp_path, monkeypatch):
    session, console = build_session(tmp_path, monkeypatch)
    asyncio.run(session.submit("/theme"))
    assert "midnight" in console.export_text()
    asyncio.run(session.submit("/theme paper"))
    assert session.renderer.theme.background == resolve("paper").background
    assert (tmp_path / "theme").read_text(encoding="utf-8").strip() == "paper"
    asyncio.run(session.submit("/theme nope"))
    assert "Unknown theme" in console.export_text()
    assert session.renderer.theme.background == resolve("paper").background

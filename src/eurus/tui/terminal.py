from __future__ import annotations

import atexit
import os
import re
import signal
from collections.abc import Mapping
from typing import Any, TextIO

from eurus.tui.theme import Theme

MODES = ("auto", "always", "never")
LAYOUTS = ("inline", "fullscreen")
CONTROL = re.compile(r"[\x00-\x1f\x7f]")
OSC = "\x1b]"
BEL = "\x07"
RESET_COLORS = f"{OSC}111{BEL}{OSC}110{BEL}{OSC}112{BEL}"
PUSH_TITLE = "\x1b[22;2t"
POP_TITLE = "\x1b[23;2t"
SIGNAL_NAMES = ("SIGTERM", "SIGHUP")
SIGNALS = tuple(getattr(signal, name) for name in SIGNAL_NAMES if hasattr(signal, name))


def recolor_mode(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    value = env.get("EURUS_RECOLOR", "auto").strip().lower()
    return value if value in MODES else "auto"


def layout_mode(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    value = env.get("EURUS_LAYOUT", "inline").strip().lower()
    return value if value in LAYOUTS else "inline"


def color_sequence(theme: Theme) -> str:
    return f"{OSC}11;{theme.background}{BEL}{OSC}10;{theme.text}{BEL}{OSC}12;{theme.copper}{BEL}"


def title_sequence(text: str) -> str:
    return f"{OSC}2;{CONTROL.sub('', text)[:80]}{BEL}"


def supported(stream: TextIO, mode: str, environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    if mode == "never":
        return False
    try:
        if not stream.isatty():
            return False
    except (OSError, ValueError):
        return False
    if mode == "always":
        return True
    term = env.get("TERM", "")
    if term in ("", "dumb") or term.startswith(("screen", "tmux")):
        return False
    return not (env.get("TMUX") or "NO_COLOR" in env)


class Recolor:
    def __init__(
        self,
        stream: TextIO,
        mode: str = "auto",
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.stream = stream
        self.enabled = supported(stream, mode, environ)
        self.active = False
        self.titled = False
        self.registered = False
        self.previous: dict[int, Any] = {}

    def _write(self, text: str) -> None:
        try:
            self.stream.write(text)
            self.stream.flush()
        except (OSError, ValueError):
            return

    def apply(self, theme: Theme, title: str = "") -> None:
        if not self.enabled:
            return
        first = not self.active
        if first and title:
            self._write(PUSH_TITLE)
            self.titled = True
        self._write(color_sequence(theme))
        if title:
            self._write(title_sequence(title))
        if first:
            self.active = True
            self._install()

    def restore(self) -> None:
        if not self.active:
            return
        self.active = False
        self._write(RESET_COLORS + (POP_TITLE if self.titled else ""))
        self.titled = False
        self._uninstall()

    def _install(self) -> None:
        if not self.registered:
            atexit.register(self.restore)
            self.registered = True
        for number in SIGNALS:
            try:
                self.previous[number] = signal.signal(number, self._handle)
            except (ValueError, OSError):
                continue

    def _uninstall(self) -> None:
        for number, handler in self.previous.items():
            try:
                signal.signal(number, handler)
            except (ValueError, OSError, TypeError):
                continue
        self.previous.clear()

    def _handle(self, number: int, frame: Any) -> None:
        self.restore()
        signal.signal(number, signal.SIG_DFL)
        os.kill(os.getpid(), number)

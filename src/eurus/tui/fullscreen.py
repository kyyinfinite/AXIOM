from __future__ import annotations

import asyncio
import contextlib
import shutil
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, cast

from prompt_toolkit import Application
from prompt_toolkit.application import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples, to_formatted_text
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import (
    CompletionsMenu,
    ConditionalContainer,
    Dimension,
    Float,
    FloatContainer,
    HSplit,
    Layout,
    VSplit,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.styles import DynamicStyle
from rich.console import Console

from eurus.tui.commands import COMMANDS
from eurus.tui.layout import clip, format_tokens
from eurus.tui.prompt import MAX_INPUT_LINES, MENU_ROWS, PromptBox, fragments
from eurus.tui.renderer import TerminalRenderer
from eurus.tui.theme import Theme
from eurus.tui.transcript import TranscriptSink

if TYPE_CHECKING:
    from eurus.tui.session import InteractiveSession

PANEL_WIDTH = 30
WIDE_COLUMNS = 100
SCROLL_STEP = 3
INTERRUPT = object()


def pane_width(columns: int) -> int:
    panel = PANEL_WIDTH if columns >= WIDE_COLUMNS else 0
    return max(columns - panel - 1, 20)


class FullscreenRenderer(TerminalRenderer):
    def __init__(self, console: Console, theme: Theme, sink: TranscriptSink) -> None:
        super().__init__(console, theme)
        self.sink = sink

    def clear(self) -> None:
        self.sink.clear()


def attach(session: InteractiveSession) -> tuple[FullscreenRenderer, TranscriptSink]:
    size = shutil.get_terminal_size((80, 24))
    sink = TranscriptSink()
    console = Console(
        file=cast(IO[str], sink),
        force_terminal=False,
        color_system="truecolor",
        width=pane_width(size.columns),
        height=size.lines,
        highlight=False,
    )
    renderer = FullscreenRenderer(console, session.renderer.theme, sink)
    session.renderer = renderer
    return renderer, sink


class FullscreenUI(PromptBox):
    def __init__(
        self, session: InteractiveSession, renderer: FullscreenRenderer, sink: TranscriptSink
    ) -> None:
        self.session = session
        self.renderer = renderer
        self.sink = sink
        self.mode = "chat"
        self.question = ""
        self.items: tuple[tuple[str, str], ...] = ()
        self.follow = True
        self.top = 0
        self.interrupts = 0
        self.task: asyncio.Task[None] | None = None
        self.future: asyncio.Future[Any] | None = None
        super().__init__(renderer.theme, self._status, self._hint, COMMANDS)
        self.buffer = Buffer(
            multiline=True,
            completer=self.completer,
            complete_while_typing=True,
            history=self.history,
        )
        self.window = Window(
            FormattedTextControl(self._render_transcript),
            wrap_lines=False,
            always_hide_cursor=True,
        )
        self.app: Application[None] = self._build()
        sink.on_change = self.app.invalidate

    def _status(self, width: int) -> list[tuple[str, str]]:
        session = self.session
        return self.renderer.status_parts(
            session.state,
            session.config.agent.model,
            Path(session.config.permissions.workspace).expanduser(),
            session.usage,
            bool(session.retry_prompt),
            width,
        )

    def _hint(self, width: int) -> list[tuple[str, str]]:
        if self.mode == "ask":
            return self.renderer.hint_parts(width, (("Enter", "answer"), ("Esc", "cancel")))
        if self.mode == "confirm":
            return self.renderer.hint_parts(width, self.items)
        return self.renderer.hint_parts(width)

    def _size(self) -> tuple[int, int]:
        size = get_app().output.get_size()
        return int(size.columns), int(size.rows)

    def _height(self) -> int:
        info = self.window.render_info
        if info is not None:
            return max(int(info.window_height), 1)
        return max(self._size()[1] - 8, 3)

    def _limit(self) -> int:
        return max(len(self.sink.lines) - self._height(), 0)

    def _scroll(self, delta: int) -> None:
        current = self._limit() if self.follow else self.top
        self.top = min(max(current + delta, 0), self._limit())
        self.follow = self.top >= self._limit()

    def _render_transcript(self) -> StyleAndTextTuples:
        columns, rows = self._size()
        console = self.renderer.console
        width = pane_width(columns)
        if console.width != width:
            console.width = width
        console.height = rows
        limit = self._limit()
        if self.follow:
            self.top = limit
        self.top = min(max(self.top, 0), limit)
        view = self.sink.lines[self.top : self.top + self._height()]
        out: StyleAndTextTuples = []
        for index, line in enumerate(view):
            out.extend(to_formatted_text(ANSI(line)))
            if index < len(view) - 1:
                out.append(("", "\n"))
        return out

    def _render_header(self) -> StyleAndTextTuples:
        config = self.session.config
        dot = self.theme.glyphs.dot
        workspace = Path(config.permissions.workspace).expanduser().name
        model = config.agent.model or "no model"
        return [
            ("class:title", "EURUS"),
            ("class:dim", f"  {dot}  {model}  {dot}  {workspace}"),
        ]

    def _render_panel(self) -> StyleAndTextTuples:
        session = self.session
        config = session.config
        ellipsis = self.renderer.ellipsis
        room = PANEL_WIDTH - 3
        rows = (
            ("MODEL", config.agent.model or "not configured"),
            ("WORKSPACE", Path(config.permissions.workspace).expanduser().name or "."),
            ("TOKENS", format_tokens(session.usage.total_tokens)),
            ("STATE", session.state.value),
        )
        out: StyleAndTextTuples = [("class:title", " SESSION\n\n")]
        for label, value in rows:
            out.append(("class:dim", f" {label}\n"))
            out.append(("class:input", f" {clip(value, room, ellipsis)}\n\n"))
        return out

    def _label(self) -> str:
        return self.question if self.mode == "ask" else self.theme.glyphs.prompt

    def _render_glyph(self) -> StyleAndTextTuples:
        return [("class:glyph", f"{self._label()} ")]

    def _render_confirm(self) -> StyleAndTextTuples:
        return [("class:warn", f" {self.question}")]

    def _build(self) -> Application[None]:
        confirming = Condition(lambda: self.mode == "confirm")
        wide = Condition(lambda: self._size()[0] >= WIDE_COLUMNS)
        side = self.chars[5]
        panel = ConditionalContainer(
            VSplit(
                [
                    Window(width=1, char=side, style="class:border"),
                    Window(FormattedTextControl(self._render_panel), width=PANEL_WIDTH - 1),
                ]
            ),
            filter=wide,
        )
        header = HSplit(
            [
                Window(FormattedTextControl(self._render_header), height=1),
                Window(height=1, char=self.chars[4], style="class:border"),
            ]
        )
        field = Window(
            BufferControl(buffer=self.buffer),
            height=Dimension(min=1, max=MAX_INPUT_LINES),
            wrap_lines=True,
            dont_extend_height=True,
            style="class:input",
        )
        glyph = Window(
            FormattedTextControl(self._render_glyph),
            width=lambda: Dimension.exact(len(self._label()) + 1),
            height=1,
            dont_extend_height=True,
        )
        box = self._frame(VSplit([glyph, field]))
        footer = HSplit(
            [
                Window(
                    FormattedTextControl(lambda: fragments(self.status(self._size()[0]))),
                    height=1,
                ),
                Window(
                    FormattedTextControl(lambda: fragments(self.hint(self._size()[0]))),
                    height=1,
                ),
            ]
        )
        bar = ConditionalContainer(
            Window(FormattedTextControl(self._render_confirm), height=1), filter=confirming
        )
        root = FloatContainer(
            content=HSplit([header, VSplit([self.window, panel]), bar, box, footer]),
            floats=[
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=CompletionsMenu(max_height=MENU_ROWS, scroll_offset=1),
                )
            ],
        )
        return Application(
            layout=Layout(root, focused_element=field),
            key_bindings=self._keys(),
            style=DynamicStyle(lambda: self.style),
            full_screen=True,
            refresh_interval=0.3,
        )

    def _keys(self) -> KeyBindings:
        keys = KeyBindings()
        confirming = Condition(lambda: self.mode == "confirm")
        asking = Condition(lambda: self.mode != "chat")
        typing = Condition(lambda: self.mode != "confirm")
        empty = Condition(lambda: not self.buffer.text)

        @keys.add(Keys.Any, filter=confirming)
        def swallow(event: KeyPressEvent) -> None:
            return None

        for key in ("y", "a", "Y", "A"):
            keys.add(key, filter=confirming, eager=True)(lambda event: self._settle(True))
        for key in ("n", "d", "N", "D"):
            keys.add(key, filter=confirming, eager=True)(lambda event: self._settle(False))

        @keys.add("enter")
        def enter(event: KeyPressEvent) -> None:
            if self.mode == "confirm":
                self._settle(False)
                return
            text = self.buffer.text
            if text.endswith("\\"):
                self.buffer.text = text[:-1] + "\n"
                self.buffer.cursor_position = len(self.buffer.text)
                return
            self.buffer.append_to_history()
            self.buffer.reset()
            if self.mode == "ask":
                self._settle(text)
            else:
                self.dispatch(text)

        @keys.add("c-j", filter=typing)
        @keys.add("escape", "enter", filter=typing)
        def newline(event: KeyPressEvent) -> None:
            self.buffer.insert_text("\n")

        @keys.add("tab", filter=typing)
        def complete(event: KeyPressEvent) -> None:
            if self.buffer.complete_state is not None:
                self.buffer.complete_next()
            else:
                self.buffer.start_completion(select_first=True)

        @keys.add("escape", filter=asking, eager=True)
        def escape(event: KeyPressEvent) -> None:
            self._cancel_prompt()

        @keys.add("c-c")
        def interrupt(event: KeyPressEvent) -> None:
            if self.mode == "confirm":
                self._settle(INTERRUPT)
            elif self.mode == "ask":
                self._cancel_prompt()
            elif self.task is not None and not self.task.done():
                self.task.cancel()
            elif self.buffer.text:
                self.buffer.reset()
                self.interrupts = 0
            else:
                self.interrupts += 1
                if self.interrupts >= 2:
                    event.app.exit()
                else:
                    self.renderer.status("Press Ctrl+C again or Ctrl+D to exit.")

        @keys.add("c-d", filter=empty)
        def eof(event: KeyPressEvent) -> None:
            if self.mode != "chat":
                self._cancel_prompt()
                return
            if self.task is not None and not self.task.done():
                self.task.cancel()
            event.app.exit()

        @keys.add("pageup")
        def page_up(event: KeyPressEvent) -> None:
            self._scroll(-max(self._height() - 2, 1))

        @keys.add("pagedown")
        def page_down(event: KeyPressEvent) -> None:
            self._scroll(max(self._height() - 2, 1))

        @keys.add("c-up")
        def line_up(event: KeyPressEvent) -> None:
            self._scroll(-SCROLL_STEP)

        @keys.add("c-down")
        def line_down(event: KeyPressEvent) -> None:
            self._scroll(SCROLL_STEP)

        return keys

    def _cancel_prompt(self) -> None:
        self.buffer.reset()
        self._settle(False if self.mode == "confirm" else None)

    def _settle(self, value: Any) -> None:
        if self.future is not None and not self.future.done():
            self.future.set_result(value)

    def dispatch(self, text: str) -> None:
        if self.task is not None and not self.task.done():
            self.renderer.status("Busy. Press Ctrl+C to cancel the current task.")
            return
        self.interrupts = 0
        self.follow = True
        self.task = asyncio.get_running_loop().create_task(self._run(text))

    async def _run(self, text: str) -> None:
        try:
            await self.session.submit(text)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.renderer.error(f"Unexpected error: {type(exc).__name__}: {exc}")
        finally:
            if not self.session.running:
                with contextlib.suppress(Exception):
                    self.app.exit()
            self.app.invalidate()

    async def _ask_user(self, mode: str, question: str, completer: WordCompleter | None) -> Any:
        self.future = asyncio.get_running_loop().create_future()
        self.mode = mode
        self.question = question
        if completer is not None:
            self.buffer.completer = completer
        self.buffer.reset()
        self.app.invalidate()
        try:
            return await self.future
        finally:
            self.mode = "chat"
            self.question = ""
            self.future = None
            self.buffer.completer = self.completer
            self.buffer.reset()
            self.app.invalidate()

    async def confirm(self, question: str, yes: str, no: str, yes_keys: Any = ("y",)) -> bool:
        self.items = ((str(yes_keys[0]), yes), ("Enter", no))
        result = await self._ask_user("confirm", question, None)
        if result is INTERRUPT:
            raise KeyboardInterrupt
        return bool(result)

    async def ask_options(self, label: str, options: list[str]) -> str | None:
        completer = WordCompleter(options, ignore_case=True, sentence=True)
        answer: str | None = await self._ask_user("ask", label, completer)
        return answer


async def run_fullscreen(session: InteractiveSession) -> None:
    renderer, sink = attach(session)
    ui = FullscreenUI(session, renderer, sink)
    session.picker = ui
    renderer.startup(session.config)
    try:
        await ui.app.run_async()
    finally:
        if ui.task is not None and not ui.task.done():
            ui.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ui.task
        await session.close()
        renderer.shutdown()

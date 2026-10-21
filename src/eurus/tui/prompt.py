from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from prompt_toolkit import Application
from prompt_toolkit.application import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
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
from prompt_toolkit.layout.containers import AnyContainer
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.styles import Style

from eurus.tui.commands import Command
from eurus.tui.theme import Theme

Parts = Callable[[int], list[tuple[str, str]]]

MAX_INPUT_LINES = 8
MENU_ROWS = 8
UNICODE_BOX = ("╭", "╮", "╰", "╯", "─", "│")
ASCII_BOX = ("+", "+", "+", "+", "-", "|")


class SlashCompleter(Completer):
    def __init__(self, commands: Iterable[Command]) -> None:
        self.commands = tuple(commands)

    def get_completions(
        self, document: Document, complete_event: CompleteEvent | None
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/") or any(char.isspace() for char in text):
            return
        key = text.lower()
        for command in self.commands:
            if command.name.startswith(key):
                yield Completion(
                    command.name,
                    start_position=-len(text),
                    display_meta=command.summary,
                )


def fragments(parts: list[tuple[str, str]]) -> StyleAndTextTuples:
    return [(f"fg:{style}", text) for text, style in parts]


def terminal_width() -> int:
    return int(get_app().output.get_size().columns)


class PromptBox:
    def __init__(
        self,
        theme: Theme,
        status: Parts,
        hint: Parts,
        commands: Iterable[Command],
    ) -> None:
        self.status = status
        self.hint = hint
        self.completer = SlashCompleter(commands)
        self.history = InMemoryHistory()
        self.apply_theme(theme)

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.chars = UNICODE_BOX if theme.unicode else ASCII_BOX
        self.style = Style.from_dict(
            {
                "border": f"fg:{theme.border}",
                "glyph": f"bold fg:{theme.copper}",
                "dim": f"fg:{theme.dim}",
                "key": f"bold fg:{theme.copper}",
                "title": f"bold fg:{theme.copper}",
                "warn": f"fg:{theme.yellow}",
                "input": f"fg:{theme.text}",
                "completion-menu": f"bg:{theme.surface}",
                "completion-menu.completion": f"fg:{theme.text} bg:{theme.surface}",
                "completion-menu.completion.current": f"fg:{theme.surface} bg:{theme.copper}",
                "completion-menu.meta.completion": f"fg:{theme.dim} bg:{theme.surface}",
                "completion-menu.meta.completion.current": (
                    f"fg:{theme.surface} bg:{theme.copper}"
                ),
            }
        )

    def _edge(self, left: str, right: str) -> VSplit:
        fill = self.chars[4]
        return VSplit(
            [
                Window(width=1, height=1, char=left, style="class:border"),
                Window(height=1, char=fill, style="class:border"),
                Window(width=1, height=1, char=right, style="class:border"),
            ],
            height=1,
        )

    def _frame(self, body: AnyContainer) -> HSplit:
        side = self.chars[5]
        middle = VSplit(
            [
                Window(width=1, char=side, style="class:border"),
                VSplit([Window(width=1), body, Window(width=1)]),
                Window(width=1, char=side, style="class:border"),
            ]
        )
        return HSplit(
            [
                self._edge(self.chars[0], self.chars[1]),
                middle,
                self._edge(self.chars[2], self.chars[3]),
            ]
        )

    def _glyph(self) -> Window:
        return Window(
            width=2,
            height=1,
            content=FormattedTextControl([("class:glyph", f"{self.theme.glyphs.prompt} ")]),
            dont_extend_height=True,
        )

    def _hint_line(self, items: Sequence[tuple[str, str]]) -> Window:
        def render() -> StyleAndTextTuples:
            width = terminal_width()
            out: StyleAndTextTuples = []
            used = 0
            for key, action in items:
                need = len(key) + 1 + len(action) + (2 if out else 0)
                if used + need > width:
                    break
                if out:
                    out.append(("class:dim", "  "))
                out.append(("class:key", key))
                out.append(("class:dim", f" {action}"))
                used += need
            return out

        return Window(FormattedTextControl(render), height=1)

    def _bindings(self, buffer: Buffer) -> KeyBindings:
        bindings = KeyBindings()
        empty = Condition(lambda: not buffer.text)

        @bindings.add("enter")
        def submit(event: KeyPressEvent) -> None:
            text = buffer.text
            if text.endswith("\\"):
                buffer.text = text[:-1] + "\n"
                buffer.cursor_position = len(buffer.text)
                return
            buffer.append_to_history()
            event.app.exit(result=text)

        @bindings.add("c-j")
        @bindings.add("escape", "enter")
        def newline(event: KeyPressEvent) -> None:
            buffer.insert_text("\n")

        @bindings.add("tab")
        def complete(event: KeyPressEvent) -> None:
            if buffer.complete_state is not None:
                buffer.complete_next()
            else:
                buffer.start_completion(select_first=True)

        @bindings.add("c-c")
        def cancel(event: KeyPressEvent) -> None:
            if buffer.text:
                buffer.reset()
            else:
                event.app.exit(exception=KeyboardInterrupt)

        @bindings.add("c-d", filter=empty)
        def eof(event: KeyPressEvent) -> None:
            event.app.exit(exception=EOFError)

        return bindings

    def _layout(self, buffer: Buffer) -> Layout:
        field = Window(
            content=BufferControl(buffer=buffer),
            height=Dimension(min=1, max=MAX_INPUT_LINES),
            wrap_lines=True,
            dont_extend_height=True,
            style="class:input",
        )
        box = self._frame(VSplit([self._glyph(), field]))
        footer = HSplit(
            [
                Window(
                    FormattedTextControl(lambda: fragments(self.status(terminal_width()))),
                    height=1,
                ),
                Window(
                    FormattedTextControl(lambda: fragments(self.hint(terminal_width()))),
                    height=1,
                ),
            ]
        )
        reserve = ConditionalContainer(
            Window(height=Dimension.exact(MENU_ROWS)),
            filter=Condition(lambda: buffer.text.startswith("/") and "\n" not in buffer.text),
        )
        root = FloatContainer(
            content=HSplit([box, footer, reserve]),
            floats=[
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=CompletionsMenu(max_height=MENU_ROWS, scroll_offset=1),
                )
            ],
        )
        return Layout(root, focused_element=field)

    async def ask(self) -> str:
        buffer = Buffer(
            multiline=True,
            completer=self.completer,
            complete_while_typing=True,
            history=self.history,
        )
        app: Application[str] = Application(
            layout=self._layout(buffer),
            key_bindings=self._bindings(buffer),
            style=self.style,
            full_screen=False,
            erase_when_done=True,
        )
        return await app.run_async()

    async def confirm(
        self, question: str, yes: str, no: str, yes_keys: Sequence[str] = ("y",)
    ) -> bool:
        body = VSplit(
            [
                self._glyph(),
                Window(
                    FormattedTextControl([("class:input", question)]),
                    height=1,
                    dont_extend_height=True,
                ),
            ]
        )
        root = HSplit([self._frame(body), self._hint_line([(yes_keys[0], yes), ("Enter", no)])])
        bindings = KeyBindings()

        def accept(event: KeyPressEvent) -> None:
            event.app.exit(result=True)

        def reject(event: KeyPressEvent) -> None:
            event.app.exit(result=False)

        def interrupt(event: KeyPressEvent) -> None:
            event.app.exit(exception=KeyboardInterrupt)

        for key in yes_keys:
            bindings.add(key)(accept)
            bindings.add(key.upper())(accept)
        for key in ("enter", "escape", "n", "d"):
            bindings.add(key, eager=key == "escape")(reject)
        bindings.add("c-c")(interrupt)
        app: Application[bool] = Application(
            layout=Layout(root),
            key_bindings=bindings,
            style=self.style,
            full_screen=False,
            erase_when_done=True,
        )
        return await app.run_async()

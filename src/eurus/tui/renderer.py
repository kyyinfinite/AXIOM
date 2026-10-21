from __future__ import annotations

import re
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from rich.box import ASCII, ROUNDED, Box
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme as RichTheme

from eurus import __version__
from eurus.agent.events import AgentEvent
from eurus.config import Config
from eurus.models import ToolCall, ToolResult, Usage
from eurus.providers.discovery import ModelInfo
from eurus.security.approval import ApprovalRequest
from eurus.tui.brand import header
from eurus.tui.capabilities import ColorLevel, TerminalCapabilities, detect
from eurus.tui.commands import Command
from eurus.tui.info import credential_note, describe, shell_label
from eurus.tui.layout import Segment, clip, fit_segments, format_tokens, shorten_path
from eurus.tui.picker import ProviderRow
from eurus.tui.redact import redact_arguments, redact_text, summarize_arguments
from eurus.tui.state import AgentState
from eurus.tui.theme import Theme
from eurus.tui.themes import PALETTE_KEYS

ColorSystem = Literal["standard", "256", "truecolor"]

COLOR_SYSTEMS: dict[ColorLevel, ColorSystem | None] = {
    "truecolor": "truecolor",
    "256": "256",
    "basic": "standard",
    "none": None,
}
SHORTCUTS: tuple[tuple[str, str], ...] = (
    ("Enter", "send"),
    ("\\", "newline"),
    ("Tab", "complete"),
    ("Ctrl+C", "cancel"),
    ("Ctrl+D", "exit"),
)
SHELL_PREVIEW_LIMIT = 1500
ALLOW_ANSWERS = frozenset({"a", "allow", "y", "yes"})


def markdown_styles(theme: Theme) -> dict[str, str]:
    return {
        "markdown.code": f"bold {theme.copper} on {theme.surface}",
        "markdown.code_block": f"{theme.text} on {theme.surface}",
        "markdown.link": f"underline {theme.amber}",
        "markdown.link_url": f"underline {theme.dim}",
        "markdown.item.bullet": f"bold {theme.copper}",
        "markdown.item.number": f"bold {theme.copper}",
        "markdown.block_quote": f"italic {theme.dim}",
        "markdown.hr": theme.border,
        "markdown.h1": f"bold {theme.text}",
        "markdown.h2": f"bold {theme.copper}",
        "markdown.h3": f"bold {theme.copper}",
        "markdown.h4": f"bold {theme.dim}",
    }


def make_console(caps: TerminalCapabilities | None = None) -> Console:
    caps = caps or detect()
    return Console(color_system=COLOR_SYSTEMS[caps.color], highlight=False)


def capabilities_of(console: Console) -> TerminalCapabilities:
    levels: dict[str, ColorLevel] = {
        "truecolor": "truecolor",
        "256": "256",
        "standard": "basic",
        "windows": "basic",
    }
    return TerminalCapabilities(
        color=levels.get(console.color_system or "", "none"),
        unicode=(console.encoding or "").lower().replace("-", "").startswith("utf"),
        width=max(console.width, 20),
    )


class StreamView:
    def __init__(self, theme: Theme) -> None:
        self.theme = theme
        self.buffer = ""
        self.state = AgentState.THINKING
        self.started = time.monotonic()

    def feed(self, text: str) -> None:
        self.buffer += text

    def take(self) -> str:
        text, self.buffer = self.buffer, ""
        return text

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        rows = max(3, min(10, console.height - 6))
        if self.buffer.strip():
            wrapped = Text(self.buffer[-4000:], style=self.theme.text).wrap(
                console, options.max_width
            )
            yield Text("\n").join(wrapped[-rows:])
        elapsed = time.monotonic() - self.started
        frames = self.theme.glyphs.frames
        frame = frames[int(elapsed * 10) % len(frames)]
        line = Text(f"{frame} ", style=self.theme.state_style(self.state))
        line.append(self.state.value.upper(), style=f"bold {self.theme.state_style(self.state)}")
        line.append(f" {elapsed:.0f}s  Ctrl+C cancel", style=self.theme.dim)
        yield line


class TerminalRenderer:
    def __init__(self, console: Console | None = None, theme: Theme | None = None) -> None:
        self.console = console or make_console()
        self.theme = theme or Theme.for_unicode(capabilities_of(self.console).unicode)
        self._styled = False
        self.apply_theme(self.theme)
        self._started: dict[str, float] = {}
        self._live: Live | None = None

    def apply_theme(self, theme: Theme) -> None:
        if self._styled:
            self.console.pop_theme()
        self.theme = theme
        self.console.push_theme(RichTheme(markdown_styles(theme)))
        self._styled = True

    @property
    def width(self) -> int:
        return max(self.console.width, 20)

    @property
    def box_style(self) -> Box:
        return ROUNDED if capabilities_of(self.console).unicode else ASCII

    @property
    def ellipsis(self) -> str:
        return self.theme.glyphs.ellipsis

    def shortcuts(self) -> Text:
        lines: list[Text] = []
        current = Text()
        for key, action in SHORTCUTS:
            item = Text()
            item.append(key, style=f"bold {self.theme.copper}")
            item.append(f" {action}", style=self.theme.dim)
            if current.cell_len and current.cell_len + 2 + item.cell_len > self.width:
                lines.append(current)
                current = Text()
            if current.cell_len:
                current.append("  ")
            current.append_text(item)
        lines.append(current)
        return Text("\n").join(lines)

    def startup(self, config: Config) -> None:
        info = describe(config)
        self.console.print(
            header(
                __version__,
                self.width,
                self.console.height,
                self.theme,
                self.console.color_system is not None,
            )
        )
        self.console.print()
        grid = Table.grid(padding=(0, 1))
        grid.add_column(style=self.theme.dim, no_wrap=True)
        grid.add_column(overflow="fold")
        room = max(self.width - 18, 10)
        grid.add_row("Model", clip(info.model, room, self.ellipsis))
        provider = (
            info.provider
            if info.credential == "not configured"
            else f"{info.provider} - {credential_note(info.credential)}"
        )
        grid.add_row("Provider", provider)
        workspace = Path(config.permissions.workspace).expanduser()
        grid.add_row("Workspace", shorten_path(workspace, room, self.ellipsis))
        grid.add_row("Skills", ", ".join(config.agent.skills) or "none")
        grid.add_row("Shell", shell_label(config))
        self.console.print(
            Panel(
                grid,
                title=Text("SESSION", style=f"bold {self.theme.copper}"),
                title_align="left",
                border_style=self.theme.border,
                box=self.box_style,
                padding=(0, 1),
            )
        )
        self.console.print()
        self.console.print(Text("Describe a task, or type / for commands.", style=self.theme.text))
        self.console.print(self.shortcuts())
        self.console.print()

    def prompt(self, continuation: bool = False) -> Text:
        glyph = self.theme.glyphs.cont if continuation else self.theme.glyphs.prompt
        return Text(f"{glyph} ", style=f"bold {self.theme.copper}")

    def ask(self, prompt: Text) -> str:
        if not self.console.is_terminal:
            return self.console.input(prompt)
        with self.console.capture() as capture:
            self.console.print(prompt, end="")
        safe = re.sub(r"(\x1b\[[0-9;]*m)", "\x01\\1\x02", capture.get())
        return input(safe)

    def status_parts(
        self,
        state: AgentState,
        model: str,
        workspace: Path,
        usage: Usage,
        retry: bool,
        width: int,
    ) -> list[tuple[str, str]]:
        dot = self.theme.glyphs.dot
        segments = [
            Segment(f"{dot} {state.value}"),
            Segment(model or "no model", flexible=True),
        ]
        if usage.total_tokens:
            segments.append(Segment(f"{format_tokens(usage.total_tokens)} tok"))
        segments.append(Segment(shorten_path(workspace, 28, self.ellipsis), flexible=True))
        if retry:
            segments.append(Segment("/retry"))
        segments.append(Segment("/help"))
        result: list[tuple[str, str]] = []
        for index, part in enumerate(fit_segments(segments, width, " | ")):
            if index:
                result.append((" | ", self.theme.border))
            style = self.theme.state_style(state) if index == 0 else self.theme.dim
            result.append((part, style))
        return result

    def hint_parts(
        self, width: int, items: Sequence[tuple[str, str]] | None = None
    ) -> list[tuple[str, str]]:
        items = items or (
            ("Enter", "send"),
            ("\\ or Ctrl+J", "newline"),
            ("Tab", "complete"),
            ("Ctrl+C", "clear"),
            ("Ctrl+D", "exit"),
        )
        result: list[tuple[str, str]] = []
        used = 0
        for key, action in items:
            need = len(key) + 1 + len(action) + (2 if result else 0)
            if used + need > width:
                break
            if result:
                result.append(("  ", self.theme.dim))
            result.append((key, self.theme.copper))
            result.append((f" {action}", self.theme.dim))
            used += need
        return result

    def status_bar(
        self, state: AgentState, model: str, workspace: Path, usage: Usage, retry: bool
    ) -> None:
        line = Text()
        for part, style in self.status_parts(state, model, workspace, usage, retry, self.width):
            line.append(part, style=style)
        self.console.print(Text(self.theme.glyphs.rule * self.width, style=self.theme.border))
        self.console.print(line)

    def menu_prompt(self, label: str) -> Text:
        return Text(f"{label} ", style=f"bold {self.theme.copper}")

    def provider_page(self, rows: Sequence[ProviderRow], active_model: str) -> None:
        grid = Table.grid(padding=(0, 1))
        grid.add_column(style=self.theme.dim, justify="right", no_wrap=True)
        grid.add_column(no_wrap=True)
        grid.add_column(no_wrap=True)
        grid.add_column(overflow="fold")
        dot = self.theme.glyphs.dot
        for number, row in enumerate(rows, 1):
            marker = Text(dot if row.active else " ", style=self.theme.green)
            name_style = f"bold {self.theme.text}" if row.ready else self.theme.dim
            note_style = self.theme.green if row.ready else self.theme.dim
            grid.add_row(
                str(number),
                marker,
                Text(row.name, style=name_style),
                Text(row.note, style=note_style),
            )
        current = clip(active_model or "none", max(self.width - 14, 10), self.ellipsis)
        body = Group(
            Text.assemble(("Active  ", self.theme.dim), (current, f"bold {self.theme.green}")),
            Text(),
            grid,
        )
        self.console.print(
            Panel(
                body,
                title=Text("PROVIDERS", style=f"bold {self.theme.copper}"),
                title_align="left",
                border_style=self.theme.border,
                box=self.box_style,
                padding=(0, 1),
            )
        )
        self.console.print(
            Text("number or name to open | q quit", style=self.theme.dim),
        )

    def model_page(
        self,
        provider: str,
        visible: Sequence[ModelInfo],
        start: int,
        page: int,
        pages: int,
        total: int,
        active_id: str,
        query: str,
        error: str,
    ) -> None:
        grid = Table.grid(padding=(0, 1), expand=True)
        grid.add_column(width=3, justify="right", style=self.theme.dim, no_wrap=True)
        grid.add_column(width=2, no_wrap=True)
        grid.add_column(ratio=1, overflow="fold")
        grid.add_column(width=14, justify="right", style=self.theme.dim, no_wrap=True)
        dot = self.theme.glyphs.dot
        for offset, info in enumerate(visible, 1):
            active = info.id == active_id
            marker = Text(dot if active else " ", style=self.theme.green)
            name = Text(info.id, style=f"bold {self.theme.green}" if active else self.theme.text)
            if info.label and info.label != info.id:
                name.append(f"\n{info.label}", style=self.theme.dim)
            context = f"{format_tokens(info.context)} ctx" if info.context else ""
            grid.add_row(str(offset), marker, name, context)
        parts: list[RenderableType] = []
        if active_id:
            parts.append(
                Text.assemble(
                    ("ACTIVE  ", self.theme.dim),
                    (active_id, f"bold {self.theme.green}"),
                )
            )
            parts.append(Text())
        if error:
            parts.append(Text(error, style=self.theme.yellow))
        if visible:
            parts.append(grid)
        elif not error:
            parts.append(Text("No models match this filter.", style=self.theme.dim))
        else:
            parts.append(Text("Type =model-id to enter one manually.", style=self.theme.dim))
        title = Text(f"{provider.upper()} MODELS", style=f"bold {self.theme.copper}")
        self.console.print(
            Panel(
                Group(*parts),
                title=title,
                title_align="left",
                subtitle=Text("EURUS MODEL CATALOG", style=self.theme.dim),
                subtitle_align="right",
                border_style=self.theme.border,
                box=self.box_style,
                padding=(0, 1),
            )
        )
        status = f"{total} models  |  page {page + 1}/{pages}"
        if query:
            status += f"  |  filter: {clip(query, 20, self.ellipsis)}"
        self.console.print(Text(status, style=self.theme.dim))
        self.console.print(
            Text(
                "Type to filter  ·  number select  ·  n/p pages  ·  "
                "=id manual  ·  r refresh  ·  b back",
                style=self.theme.dim,
            )
        )

    def user(self, prompt: str) -> None:
        self.console.print()
        grid = Table.grid(padding=(0, 1), expand=True)
        grid.add_column(no_wrap=True, width=1)
        grid.add_column(ratio=1, overflow="fold")
        grid.add_row(
            Text(self.theme.glyphs.prompt, style=f"bold {self.theme.copper}"),
            Text(prompt, style=self.theme.text),
        )
        self.console.print(Padding(grid, (0, 1), style=f"on {self.theme.surface}"))

    def assistant(self, content: str, label: bool = True) -> None:
        self.console.print()
        grid = Table.grid(padding=(0, 1), expand=True)
        grid.add_column(no_wrap=True, width=1)
        grid.add_column(ratio=1)
        mark = Text(self.theme.glyphs.spark if label else " ", style=f"bold {self.theme.copper}")
        grid.add_row(mark, Markdown(content, code_theme=self.theme.code_theme, hyperlinks=True))
        self.console.print(grid)

    def system(self, message: str) -> None:
        self.console.print(Text(message, style=self.theme.dim))

    def status(self, message: str) -> None:
        self.system(message)

    def warning(self, message: str) -> None:
        line = Text("WARN ", style=f"bold {self.theme.yellow}")
        line.append(message, style=self.theme.yellow)
        self.console.print(line)

    def error(self, message: str, detail: str = "") -> None:
        body = Text(message, style=self.theme.text)
        if detail:
            body.append("\n" + detail, style=self.theme.dim)
        self.console.print(
            Panel(
                body,
                title=Text("ERROR", style=f"bold {self.theme.red}"),
                title_align="left",
                border_style=self.theme.red,
                box=self.box_style,
                padding=(0, 1),
            )
        )

    def table(self, title: str, rows: Sequence[tuple[str, str]]) -> None:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style=f"bold {self.theme.copper}", no_wrap=True)
        grid.add_column(overflow="fold")
        for key, value in rows:
            grid.add_row(key, value)
        self.console.print(
            Panel(
                grid,
                title=Text(title, style=f"bold {self.theme.text}"),
                title_align="left",
                border_style=self.theme.border,
                box=self.box_style,
                padding=(0, 1),
            )
        )

    def theme_preview(self, name: str, theme: Theme) -> None:
        previous = self.theme
        self.apply_theme(theme)
        try:
            glyphs = theme.glyphs
            swatches = Table.grid(padding=(0, 2))
            pair = self.width >= 56
            for _ in range(2 if pair else 1):
                swatches.add_column(no_wrap=True)
            cells = [
                Text.assemble(
                    ("\u2588\u2588" if theme.unicode else "##", getattr(theme, key)),
                    (f" {key:<10}", theme.dim),
                    (getattr(theme, key), theme.text),
                )
                for key in PALETTE_KEYS
            ]
            step = 2 if pair else 1
            for index in range(0, len(cells), step):
                swatches.add_row(*cells[index : index + step])
            running = Text(f"  {glyphs.arrow} ", style=theme.steel)
            running.append("RUNNING ", style=f"bold {theme.steel}")
            running.append("pytest -q", style=f"bold {theme.text}")
            results = Text()
            results.append(f"{glyphs.ok} 12 passed", style=theme.green)
            results.append("   ")
            results.append(f"{glyphs.fail} 1 failed", style=theme.red)
            self.console.print(
                Panel(
                    swatches,
                    title=Text(f"PREVIEW {name}", style=f"bold {theme.copper}"),
                    title_align="left",
                    subtitle=Text(f"code: {theme.code_theme}", style=theme.dim),
                    border_style=theme.border,
                    box=self.box_style,
                    padding=(0, 1),
                )
            )
            self.user("Refactor the parser and run the tests")
            self.assistant(
                "Updated `parser.py`:\n\n```python\ndef parse(text: str) -> list[str]:\n"
                "    return text.split()\n```"
            )
            self.console.print(running)
            self.console.print(results)
            self.warning("Sample warning line")
        finally:
            self.apply_theme(previous)

    def palette(self, commands: Sequence[Command]) -> None:
        self.table("Commands", [(command.name, command.summary) for command in commands])

    def tool_started(self, call: ToolCall) -> None:
        self._started[call.id] = time.monotonic()
        line = Text(f"  {self.theme.glyphs.arrow} ", style=self.theme.steel)
        line.append("RUNNING ", style=f"bold {self.theme.steel}")
        line.append(call.name, style=f"bold {self.theme.text}")
        self.console.print(line)
        summary = clip(summarize_arguments(call.arguments), self.width * 2 - 8, self.ellipsis)
        if summary:
            self.console.print(Text("    " + summary, style=self.theme.dim))

    def tool_finished(self, result: ToolResult) -> None:
        started = self._started.pop(result.call_id, None)
        elapsed = f" {time.monotonic() - started:.1f}s" if started is not None else ""
        failed = result.is_error
        glyph = self.theme.glyphs.fail if failed else self.theme.glyphs.ok
        style = self.theme.red if failed else self.theme.green
        line = Text(f"  {glyph} ", style=style)
        line.append("ERROR " if failed else "DONE ", style=f"bold {style}")
        line.append(result.name, style=f"bold {self.theme.text}")
        line.append(elapsed, style=self.theme.dim)
        self.console.print(line)
        detail = clip(redact_text(result.output), self.width * 2 - 8, self.ellipsis)
        if detail:
            self.console.print(Text("    " + detail, style=self.theme.dim))

    def activity(self, event: AgentEvent) -> None:
        if event.kind == "tool_call" and event.tool_call is not None:
            self.tool_started(event.tool_call)
        elif event.kind == "tool_result" and event.tool_result is not None:
            self.tool_finished(event.tool_result)
        elif event.kind == "done" and event.stop_reason in {"max_steps", "iteration_budget"}:
            self.warning("Agent iteration budget reached.")
            if event.text:
                self.system(clip(event.text, self.width * 2, self.ellipsis))
        elif event.kind == "done" and event.stop_reason == "interrupted":
            self.warning("Agent request interrupted; partial response was not added to history.")
        elif event.kind == "done" and event.stop_reason == "timeout":
            self.warning("Model request timed out; partial response was not added to history.")
        elif event.kind == "lesson" and event.text:
            self.system("Lesson saved: " + clip(event.text, self.width * 2, self.ellipsis))
        elif event.kind == "skill" and event.text:
            self.system(
                "Skill draft quarantined: " + clip(event.text, self.width * 2, self.ellipsis)
            )

    def approval(
        self, request: ApprovalRequest, arguments: dict[str, object], note: str = ""
    ) -> None:
        safe = redact_arguments(arguments)
        body = Text()
        body.append("Action  ", style=self.theme.dim)
        body.append(request.tool_name, style=f"bold {self.theme.text}")
        body.append("\nRisk    ", style=self.theme.dim)
        risk_style = self.theme.red if request.risk_level == "high" else self.theme.yellow
        body.append(request.risk_level.upper(), style=f"bold {risk_style}")
        command = safe.get("command")
        if isinstance(command, str):
            preview = command[:SHELL_PREVIEW_LIMIT]
            body.append("\nCommand\n", style=self.theme.dim)
            body.append(preview, style=self.theme.text)
            if len(command) > SHELL_PREVIEW_LIMIT:
                body.append(
                    f"\nTRUNCATED ({len(command)} chars). Deny if unsure.",
                    style=f"bold {self.theme.red}",
                )
        else:
            body.append("\nTarget\n", style=self.theme.dim)
            body.append(
                clip(summarize_arguments(arguments), SHELL_PREVIEW_LIMIT, self.ellipsis),
                style=self.theme.text,
            )
        if note:
            body.append("\n" + note, style=f"bold {self.theme.yellow}")
        body.append("\n\n")
        body.append("a", style=f"bold {self.theme.green}")
        body.append(" allow once   ", style=self.theme.dim)
        body.append("d", style=f"bold {self.theme.red}")
        body.append(" deny (default)", style=self.theme.dim)
        self.console.print(
            Panel(
                body,
                title=Text("APPROVAL", style=f"bold {self.theme.amber}"),
                title_align="left",
                border_style=self.theme.amber,
                box=self.box_style,
                padding=(0, 1),
            )
        )

    def approval_prompt(self) -> Text:
        return Text("Allow? [a/D] ", style=f"bold {self.theme.amber}")

    def approval_result(self, allowed: bool) -> None:
        if allowed:
            self.console.print(Text("Allowed once.", style=self.theme.green))
        else:
            self.console.print(Text("Denied.", style=self.theme.red))

    def cancelled(self) -> None:
        self.console.print()
        self.warning("Cancelled. Use /retry to resend the same prompt.")

    def turn_summary(self, seconds: float, tool_calls: int, usage: Usage) -> None:
        line = Text("  DONE ", style=f"bold {self.theme.green}")
        details = [f"{seconds:.1f}s", f"{tool_calls} tool{'s' if tool_calls != 1 else ''}"]
        if usage.total_tokens:
            details.append(f"{format_tokens(usage.total_tokens)} tok")
        line.append(" | ".join(details), style=self.theme.dim)
        self.console.print(line)

    @contextmanager
    def generating(self) -> Iterator[StreamView]:
        view = StreamView(self.theme)
        live = Live(
            view,
            console=self.console,
            refresh_per_second=8,
            transient=True,
            vertical_overflow="crop",
        )
        self._live = live
        live.start()
        try:
            yield view
        finally:
            live.stop()
            self._live = None

    def pause_live(self) -> None:
        if self._live is not None:
            self._live.stop()

    def resume_live(self) -> None:
        if self._live is not None:
            self._live.start()

    def clear(self) -> None:
        self.console.clear()

    def shutdown(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        self.console.show_cursor(True)
        if self.console.is_terminal:
            self.console.file.write("\x1b[0m")
            self.console.file.flush()

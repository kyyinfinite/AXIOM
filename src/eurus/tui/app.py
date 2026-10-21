from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

from eurus.config import Config
from eurus.tui.commands import COMMANDS
from eurus.tui.session import InteractiveSession

if TYPE_CHECKING:
    from eurus.tui.prompt import PromptBox


def install_completion() -> None:
    try:
        import readline
    except ImportError:
        return
    names = [command.name for command in COMMANDS]

    def complete(text: str, state: int) -> str | None:
        options = [name for name in names if name.startswith(text)] if text.startswith("/") else []
        return options[state] if state < len(options) else None

    with contextlib.suppress(Exception):
        readline.set_completer(complete)
        readline.set_completer_delims(" \t\n")
        readline.parse_and_bind("tab: complete")


def read_prompt(session: InteractiveSession) -> str:
    renderer = session.renderer
    lines: list[str] = []
    while True:
        line = renderer.ask(renderer.prompt(continuation=bool(lines)))
        if line.endswith("\\"):
            lines.append(line[:-1])
            continue
        lines.append(line)
        return "\n".join(lines)


def make_prompt_box(session: InteractiveSession) -> PromptBox | None:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    try:
        from eurus.tui.prompt import PromptBox
    except ImportError:
        return None
    renderer = session.renderer

    def status(width: int) -> list[tuple[str, str]]:
        return renderer.status_parts(
            session.state,
            session.config.agent.model,
            Path(session.config.permissions.workspace).expanduser(),
            session.usage,
            bool(session.retry_prompt),
            width,
        )

    return PromptBox(renderer.theme, status, renderer.hint_parts, COMMANDS)


async def run_interactive(session: InteractiveSession) -> None:
    renderer = session.renderer
    install_completion()
    renderer.startup(session.config)
    box = make_prompt_box(session)
    interrupts = 0
    try:
        while session.running:
            if box is None:
                renderer.status_bar(
                    session.state,
                    session.config.agent.model,
                    Path(session.config.permissions.workspace).expanduser(),
                    session.usage,
                    bool(session.retry_prompt),
                )
            else:
                renderer.console.print()
            try:
                prompt = await box.ask() if box is not None else read_prompt(session)
            except EOFError:
                renderer.console.print()
                renderer.status("Goodbye.")
                break
            except KeyboardInterrupt:
                renderer.console.print()
                interrupts += 1
                if interrupts >= 2:
                    renderer.status("Goodbye.")
                    break
                renderer.status("Press Ctrl+C again or Ctrl+D to exit.")
                continue
            except Exception:
                if box is None:
                    raise
                box = None
                renderer.warning("Input box failed; falling back to the plain prompt.")
                continue
            interrupts = 0
            await session.run_cancellable(prompt)
    finally:
        await session.close()
        renderer.shutdown()


async def chat_async(ctx: click.Context) -> None:
    config: Config = ctx.find_root().obj.config
    if not sys.stdin.isatty():
        raise click.ClickException("eurus tui needs an interactive terminal; use `eurus run`.")
    await run_interactive(InteractiveSession(config))


def run_tui(ctx: click.Context) -> None:
    asyncio.run(chat_async(ctx))

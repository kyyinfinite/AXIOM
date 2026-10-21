from pathlib import Path

import pytest
from rich.console import Console

from eurus.models import Usage
from eurus.tui.renderer import TerminalRenderer
from eurus.tui.state import AgentState


@pytest.mark.parametrize("width", [20, 40, 60, 100])
def test_status_and_hint_parts_fit_width(width):
    renderer = TerminalRenderer(Console(record=True, width=width, force_terminal=False))
    status = renderer.status_parts(
        AgentState.IDLE, "groq/" + "x" * 80, Path("/tmp/a"), Usage(5000, 100), True, width
    )
    assert sum(len(text) for text, _ in status) <= width
    assert sum(len(text) for text, _ in renderer.hint_parts(width)) <= width


def test_user_and_assistant_blocks_render():
    console = Console(record=True, width=40, force_terminal=False)
    renderer = TerminalRenderer(console)
    renderer.user("hello there")
    renderer.assistant("**done**\n\n- one\n- two")
    output = console.export_text()
    assert "hello there" in output and "done" in output and "one" in output
    assert all(len(line) <= 40 for line in output.splitlines())


def test_slash_completer_matches_prefix():
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.document import Document

    from eurus.tui.commands import COMMANDS
    from eurus.tui.prompt import SlashCompleter

    completer = SlashCompleter(COMMANDS)
    names = [item.text for item in completer.get_completions(Document("/mo"), None)]
    assert names == ["/model", "/models"]
    assert list(completer.get_completions(Document("hello"), None)) == []
    assert list(completer.get_completions(Document("/model x"), None)) == []

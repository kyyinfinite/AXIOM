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


def test_markdown_styles_use_theme_palette():
    from eurus.tui.renderer import markdown_styles
    from eurus.tui.theme import Theme

    theme = Theme()
    styles = markdown_styles(theme)
    assert theme.copper in styles["markdown.code"]
    assert all("blue" not in value and "cyan" not in value for value in styles.values())
    assert theme.teal not in theme.steel and theme.steel != "#8fb4cf"


def test_model_page_numbers_match_selection_on_later_pages():
    from eurus.providers.discovery import ModelInfo

    console = Console(record=True, width=70, force_terminal=False)
    renderer = TerminalRenderer(console)
    renderer.model_page("groq", [ModelInfo("alpha"), ModelInfo("beta")], 12, 1, 2, 14, "", "", "")
    rows = [line for line in console.export_text().splitlines() if line.startswith("│")]
    assert [row.split()[1] for row in rows] == ["1", "2"]
    assert "alpha" in rows[0] and "beta" in rows[1]


def run_confirm(keys, yes_keys=("y",)):
    import asyncio

    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from eurus.tui.commands import COMMANDS
    from eurus.tui.prompt import PromptBox
    from eurus.tui.theme import Theme

    box = PromptBox(Theme(), lambda width: [], lambda width: [], COMMANDS)

    async def go():
        with create_pipe_input() as pipe:
            pipe.send_text(keys)
            with create_app_session(input=pipe, output=DummyOutput()):
                return await box.confirm("Allow once?", "allow once", "deny", yes_keys)

    return asyncio.run(go())


def test_confirm_keys():
    assert run_confirm("a", ("a", "y")) is True
    assert run_confirm("Y", ("a", "y")) is True
    assert run_confirm("n") is False
    assert run_confirm("\r") is False
    with pytest.raises(KeyboardInterrupt):
        run_confirm("\x03")

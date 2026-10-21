import asyncio

import pytest

from eurus.config import load_config
from eurus.tui.terminal import layout_mode
from eurus.tui.transcript import KEEP_LINES, MAX_LINES, TranscriptSink


def test_sink_splits_lines_and_keeps_partial_tail():
    sink = TranscriptSink()
    sink.write("one\ntw")
    sink.write("o\nthree")
    assert sink.lines == ["one", "two", "three"]
    sink.write("\n")
    assert sink.lines[-1] == ""


def test_sink_strips_osc_private_modes_and_carriage_returns():
    sink = TranscriptSink()
    sink.write("\x1b]8;;http://x\x1b\\link\x1b]8;;\x1b\\ \x1b[?25lok\r\n")
    assert sink.lines[0] == "link ok"
    sink.write("\x1b]2;title\x07done")
    assert sink.lines[-1] == "done"


def test_sink_notifies_clears_and_caps():
    sink = TranscriptSink()
    seen = []
    sink.on_change = lambda: seen.append(sink.version)
    sink.write("a\n")
    assert seen == [1]
    sink.clear()
    assert sink.lines == [""] and seen == [1, 2]
    sink.write("x\n" * (MAX_LINES + 10))
    assert len(sink.lines) <= MAX_LINES and len(sink.lines) >= KEEP_LINES
    assert not sink.isatty() and sink.encoding == "utf-8"


def test_layout_mode_defaults_to_inline():
    assert layout_mode({}) == "inline"
    assert layout_mode({"EURUS_LAYOUT": "FullScreen"}) == "fullscreen"
    assert layout_mode({"EURUS_LAYOUT": "tiles"}) == "inline"


def build_ui(tmp_path, monkeypatch):
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from eurus.tui.fullscreen import FullscreenUI, attach
    from eurus.tui.session import InteractiveSession

    monkeypatch.setenv("EURUS_HOME", str(tmp_path))
    monkeypatch.delenv("EURUS_THEME", raising=False)
    config = load_config(cwd=tmp_path, environ={"EURUS_HOME": str(tmp_path)})
    session = InteractiveSession(config)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        renderer, sink = attach(session)
        ui = FullscreenUI(session, renderer, sink)
    session.picker = ui
    return ui, sink, session


def text_of(fragments):
    return "".join(part[1] for part in fragments)


def test_transcript_follows_bottom_and_scrolls(tmp_path, monkeypatch):
    ui, sink, _ = build_ui(tmp_path, monkeypatch)
    for number in range(100):
        sink.write(f"line {number}\n")
    rendered = text_of(ui._render_transcript())
    assert "line 99" in rendered and "line 0\n" not in rendered
    ui._scroll(-30)
    assert not ui.follow
    assert "line 99" not in text_of(ui._render_transcript())
    ui._scroll(10_000)
    assert ui.follow and "line 99" in text_of(ui._render_transcript())


def test_dispatch_runs_command_into_transcript(tmp_path, monkeypatch):
    ui, sink, _ = build_ui(tmp_path, monkeypatch)

    async def go():
        ui.dispatch("/theme")
        await ui.task

    asyncio.run(go())
    assert "midnight" in "\n".join(sink.lines)


def test_exit_command_stops_session(tmp_path, monkeypatch):
    ui, _, session = build_ui(tmp_path, monkeypatch)

    async def go():
        ui.dispatch("/exit")
        await ui.task

    asyncio.run(go())
    assert not session.running


def test_confirm_and_ask_settle_with_user_answers(tmp_path, monkeypatch):
    from eurus.tui.fullscreen import INTERRUPT

    ui, _, _ = build_ui(tmp_path, monkeypatch)

    async def yes():
        asyncio.get_running_loop().call_soon(ui._settle, True)
        return await ui.confirm("Allow once?", "allow", "deny", ("a", "y"))

    async def interrupted():
        asyncio.get_running_loop().call_soon(ui._settle, INTERRUPT)
        try:
            await ui.confirm("Allow once?", "allow", "deny", ("a", "y"))
        except KeyboardInterrupt:
            return "interrupted"
        return "no"

    async def answer():
        asyncio.get_running_loop().call_soon(ui._settle, "groq")
        return await ui.ask_options("provider>", ["groq", "openai"])

    assert asyncio.run(yes()) is True
    assert asyncio.run(interrupted()) == "interrupted"
    assert asyncio.run(answer()) == "groq"
    assert ui.mode == "chat"

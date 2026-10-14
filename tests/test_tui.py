import asyncio

from rich.console import Console

from axiom.agent.events import AgentEvent
from axiom.config import load_config
from axiom.models import ToolCall, ToolResult
from axiom.tui import InteractiveSession, TerminalRenderer


def make_session(tmp_path, model=""):
    config = load_config(
        cwd=tmp_path,
        environ={"AXIOM_HOME": str(tmp_path)},
        overrides={"agent": {"model": model}},
    )
    console = Console(record=True, width=40, force_terminal=False)
    renderer = TerminalRenderer(console)
    return InteractiveSession(config, renderer), console


def test_startup_and_narrow_terminal(tmp_path):
    session, console = make_session(tmp_path)
    session.renderer.startup(session.config)
    output = console.export_text()
    assert "AXIOM" in output and "Workspace" in output and "Ctrl+C" in output


def test_empty_input_and_unknown_command(tmp_path):
    session, console = make_session(tmp_path)
    asyncio.run(session.submit("  "))
    asyncio.run(session.submit("/unknown"))
    output = console.export_text()
    assert "Input is empty" in output and "Unknown command" in output


def test_help_clear_status_tools_exit(tmp_path):
    session, console = make_session(tmp_path)
    for command in ("/help", "/status", "/tools", "/workspace", "/model", "/models", "/compact"):
        asyncio.run(session.submit(command))
    assert "Commands" in console.export_text()
    session.history.clear()
    asyncio.run(session.submit("/clear"))
    assert session.history == []
    asyncio.run(session.submit("/exit"))
    assert not session.running


def test_tool_argument_secrets_redacted(tmp_path):
    session, console = make_session(tmp_path)
    call = ToolCall("c1", "demo", {"api_key": "super-secret", "path": "a.py"})
    session.renderer.activity(AgentEvent.tool_started(call))
    output = console.export_text()
    assert "super-secret" not in output and "REDACTED" in output


def test_missing_model_is_friendly(tmp_path):
    session, console = make_session(tmp_path)
    asyncio.run(session.submit("help me"))
    assert "No model configured" in console.export_text()


def test_tool_success_and_failure_rendering(tmp_path):
    session, console = make_session(tmp_path)
    session.renderer.activity(
        AgentEvent.tool_finished(ToolResult("c1", "workspace_list", "file a.py"))
    )
    session.renderer.activity(
        AgentEvent.tool_finished(ToolResult("c2", "workspace_read", "Tool error: missing", True))
    )
    output = console.export_text()
    assert "workspace_list" in output and "Tool error" in output


def test_user_and_assistant_render_long_content(tmp_path):
    session, console = make_session(tmp_path)
    session.renderer.user("task " * 120)
    session.renderer.assistant("# Result\n\n" + "long response " * 100)
    output = console.export_text()
    assert "YOU" in output and "AXIOM" in output and "Result" in output

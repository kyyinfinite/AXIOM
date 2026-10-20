import asyncio

import pytest
from rich.console import Console

from eurus.agent.events import AgentEvent
from eurus.config import load_config
from eurus.models import Message, ToolCall, ToolResult, Usage
from eurus.tools import Tool
from eurus.tui import InteractiveSession, TerminalRenderer


def make_session(tmp_path, model=""):
    config = load_config(
        cwd=tmp_path,
        environ={"EURUS_HOME": str(tmp_path)},
        overrides={"agent": {"model": model}},
    )
    console = Console(record=True, width=40, force_terminal=False)
    renderer = TerminalRenderer(console)
    return InteractiveSession(config, renderer), console


def test_startup_and_narrow_terminal(tmp_path):
    session, console = make_session(tmp_path)
    session.renderer.startup(session.config)
    output = console.export_text()
    assert "EURUS" in output and "Workspace" in output and "Ctrl+C" in output


def test_empty_input_and_unknown_command(tmp_path):
    session, console = make_session(tmp_path)
    asyncio.run(session.submit("  "))
    asyncio.run(session.submit("/unknown"))
    output = console.export_text()
    assert "Input is empty" in output and "Unknown command" in output


def test_help_clear_status_tools_exit(tmp_path):
    session, console = make_session(tmp_path)
    for command in ("/help", "/status", "/tools", "/workspace", "/models", "/compact"):
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
    assert "YOU" in output and "Eurus" in output and "Result" in output


class FakeRouter:
    async def aclose(self) -> None:
        return None


class FakeAgent:
    def __init__(self, approver, mode):
        self.approver = approver
        self.mode = mode
        self.history = [Message.system("system")]

    async def run(self, text):
        self.history.append(Message.user(text))
        yield AgentEvent.step_started(1)
        yield AgentEvent.text_delta("Checking files")
        if self.mode == "hang":
            await asyncio.sleep(30)
        call = ToolCall("c1", "shell", {"command": "rm -rf build", "api_key": "sk-abcdefghijk"})
        tool = Tool("shell", "run", {}, None, mutates=True, requires_approval=True)
        allowed = await self.approver(call, tool)
        yield AgentEvent.tool_started(call)
        output = "ok token=ghp_abcdefghijklmnopqrstuvwx" if allowed else "error: the user denied"
        yield AgentEvent.tool_finished(ToolResult("c1", "shell", output, not allowed))
        yield AgentEvent.text_delta("All done.")
        self.history.append(Message.assistant("All done."))
        yield AgentEvent.finished("completed", Usage(1200, 300))


def make_fake_session(tmp_path, monkeypatch, mode="run", answers=()):
    session, console = make_session(tmp_path, model="groq/llama")
    queue = list(answers)
    session.renderer.ask = lambda prompt: queue.pop(0)
    monkeypatch.setattr("eurus.tui.session.build_router", lambda config: FakeRouter())
    monkeypatch.setattr(
        "eurus.tui.session.build_agent",
        lambda config, router, **kwargs: FakeAgent(kwargs["approver"], mode),
    )
    return session, console


def test_approval_allow_runs_tool_and_redacts(tmp_path, monkeypatch):
    session, console = make_fake_session(tmp_path, monkeypatch, answers=["a"])
    asyncio.run(session.submit("clean build"))
    output = console.export_text()
    assert "APPROVAL" in output and "rm -rf build" in output and "Allowed once." in output
    assert "RUNNING" in output and "DONE" in output
    assert "sk-abcdefghijk" not in output and "ghp_" not in output
    assert session.state.value == "idle" and session.usage.total_tokens == 1500
    assert len(session.history) == 2


def test_approval_default_is_deny(tmp_path, monkeypatch):
    session, console = make_fake_session(tmp_path, monkeypatch, answers=[""])
    asyncio.run(session.submit("clean build"))
    output = console.export_text()
    assert "Denied." in output and "ERROR" in output and "Allowed once." not in output


def test_cancel_marks_retry_and_returns_idle(tmp_path, monkeypatch):
    session, console = make_fake_session(tmp_path, monkeypatch, mode="hang")

    async def scenario():
        task = asyncio.create_task(session.submit("slow task"))
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return True
        return False

    assert asyncio.run(scenario())
    assert session.retry_prompt == "slow task"
    assert session.state.value == "idle"
    assert "Cancelled" in console.export_text()


def test_ctrl_c_at_approval_denies_and_cancels(tmp_path, monkeypatch):
    session, console = make_fake_session(tmp_path, monkeypatch)

    def interrupted(prompt):
        raise KeyboardInterrupt

    session.renderer.ask = interrupted
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(session.submit("clean build"))
    assert "Denied." in console.export_text()
    assert session.retry_prompt == "clean build"


def test_slash_commands_use_real_data(tmp_path):
    session, console = make_session(tmp_path, model="groq/llama")
    for command in ("/provider", "/skills", "/config", "/status", "/models"):
        asyncio.run(session.submit(command))
    output = console.export_text()
    assert "antislop" in output and "permission mode" in output and "groq" in output
    assert "connected" not in output.replace("not verified", "")


def test_model_switch_validates_provider(tmp_path):
    session, console = make_session(tmp_path, model="groq/llama")
    asyncio.run(session.submit("/model nope/x"))
    assert session.config.agent.model == "groq/llama"
    asyncio.run(session.submit("/model openrouter/some-model"))
    assert session.config.agent.model == "openrouter/some-model"
    assert "Unknown provider" in console.export_text()


def test_ambiguous_prefix_shows_palette(tmp_path):
    session, console = make_session(tmp_path)
    asyncio.run(session.submit("/mo"))
    output = console.export_text()
    assert "/model" in output and "/models" in output and "Unknown command" not in output


@pytest.mark.parametrize("width", [40, 60, 80])
def test_status_bar_fits_width(tmp_path, width):
    session, console = make_session(tmp_path, model="openrouter/" + "x" * 80)
    console.width = width
    session.renderer.status_bar(
        session.state, session.config.agent.model, tmp_path, Usage(5000, 100), True
    )
    lines = console.export_text().splitlines()
    assert lines and all(len(line) <= width for line in lines)


def test_iteration_budget_summary_is_rendered(tmp_path):
    session, console = make_session(tmp_path)
    session.renderer.activity(
        AgentEvent.finished(
            "iteration_budget",
            Usage(),
            "Iteration budget exhausted after 500 iterations.\ncalculate: 42",
        )
    )
    output = console.export_text()
    assert "iteration budget reached" in output
    assert "calculate: 42" in output

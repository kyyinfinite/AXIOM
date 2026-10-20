import json

from click.testing import CliRunner

from eurus import __version__
from eurus import cli as cli_module
from eurus.cli import main
from eurus.providers.testing import FakeTransport, openai_stream
from eurus.providers.transport import HttpError


def run(tmp_path, *args, **env):
    return CliRunner().invoke(main, list(args), env={"EURUS_HOME": str(tmp_path), **env})


def test_version(tmp_path):
    result = run(tmp_path, "--version")
    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_subcommand_shows_help(tmp_path):
    result = run(tmp_path)
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_doctor(tmp_path):
    result = run(tmp_path, "doctor")
    assert result.exit_code == 0
    assert "eurus doctor" in result.output


def test_doctor_with_model_override(tmp_path):
    result = run(tmp_path, "--model", "groq/x", "doctor")
    assert result.exit_code == 0


def test_invalid_model_override_fails(tmp_path):
    result = run(tmp_path, "--model", "bad", "doctor")
    assert result.exit_code != 0
    assert "provider/model" in result.output


def test_providers_lists_builtin(tmp_path):
    result = run(tmp_path, "providers")
    assert result.exit_code == 0
    assert "groq" in result.output


def test_config_show_is_json(tmp_path):
    result = run(tmp_path, "config", "show")
    assert result.exit_code == 0
    assert json.loads(result.stdout)["agent"]["max_steps"] == 40


def test_config_path(tmp_path):
    result = run(tmp_path, "config", "path")
    assert result.exit_code == 0
    assert "user:" in result.output


def test_config_init_refuses_overwrite(tmp_path):
    first = run(tmp_path, "config", "init")
    assert first.exit_code == 0
    assert (tmp_path / "config.toml").is_file()
    assert run(tmp_path, "config", "init").exit_code != 0
    assert run(tmp_path, "config", "init", "--force").exit_code == 0


def test_broken_user_config_reports_error(tmp_path):
    (tmp_path / "config.toml").write_text("garbage ===", encoding="utf-8")
    result = run(tmp_path, "providers")
    assert result.exit_code != 0
    assert "invalid TOML" in result.output


def patched_router(monkeypatch, transport):
    real = cli_module.build_router
    monkeypatch.setattr(
        cli_module,
        "build_router",
        lambda config: real(config, transport=transport, environ={"GROQ_API_KEY": "k"}),
    )


def test_ask_streams_text(tmp_path, monkeypatch):
    chunk = json.dumps({"choices": [{"delta": {"content": "hello there"}}]})
    transport = FakeTransport(streams=[[chunk, "[DONE]"]])
    patched_router(monkeypatch, transport)
    result = run(tmp_path, "--model", "groq/x", "ask", "say", "hi")
    assert result.exit_code == 0
    assert "hello there" in result.output
    assert transport.calls[0].payload["messages"] == [{"role": "user", "content": "say hi"}]


def test_ask_without_stream_and_system_prompt(tmp_path, monkeypatch):
    body = {"choices": [{"message": {"content": "plain answer"}, "finish_reason": "stop"}]}
    transport = FakeTransport(json_responses=[body])
    patched_router(monkeypatch, transport)
    result = run(tmp_path, "--model", "groq/x", "ask", "--no-stream", "--system", "terse", "q")
    assert result.exit_code == 0
    assert "plain answer" in result.output
    assert transport.calls[0].payload["messages"][0] == {"role": "system", "content": "terse"}


def test_ask_requires_a_model(tmp_path):
    result = run(tmp_path, "ask", "hello")
    assert result.exit_code != 0
    assert "no model configured" in result.output


def test_ask_reports_provider_errors(tmp_path, monkeypatch):
    transport = FakeTransport(json_responses=[HttpError(401, '{"error": {"message": "bad key"}}')])
    patched_router(monkeypatch, transport)
    result = run(tmp_path, "--model", "groq/x", "ask", "--no-stream", "hello")
    assert result.exit_code != 0
    assert "bad key" in result.output


def test_tools_command_lists_builtin(tmp_path):
    result = run(tmp_path, "tools")
    assert result.exit_code == 0
    assert "calculate" in result.output
    assert "get_time" in result.output
    assert "git_status" in result.output


def test_run_executes_tools(tmp_path, monkeypatch):
    transport = FakeTransport(
        streams=[
            openai_stream("", [("c1", "calculate", {"expression": "6*7"})]),
            openai_stream("the answer is 42"),
        ]
    )
    patched_router(monkeypatch, transport)
    result = run(tmp_path, "--model", "groq/x", "run", "what", "is", "6*7")
    assert result.exit_code == 0
    assert "-> calculate" in result.output
    assert "<- ok: 42" in result.output
    assert "the answer is 42" in result.output


def test_run_reports_provider_errors(tmp_path, monkeypatch):
    transport = FakeTransport(streams=[HttpError(401, '{"error": {"message": "bad key"}}')])
    patched_router(monkeypatch, transport)
    result = run(tmp_path, "--model", "groq/x", "run", "hello")
    assert result.exit_code != 0
    assert "bad key" in result.output


def test_chat_session(tmp_path, monkeypatch):
    transport = FakeTransport(streams=[openai_stream("hi back")])
    patched_router(monkeypatch, transport)
    result = CliRunner().invoke(
        main,
        ["--model", "groq/x", "chat"],
        input="hello\n/tools\n/usage\n/clear\n/exit\n",
        env={"EURUS_HOME": str(tmp_path)},
    )
    assert result.exit_code == 0
    assert "hi back" in result.output
    assert "calculate" in result.output
    assert "get_time" in result.output
    assert "apply_patch" in result.output
    assert "conversation cleared" in result.output
    assert len(transport.calls) == 1


def test_chat_survives_provider_errors(tmp_path, monkeypatch):
    transport = FakeTransport(
        streams=[HttpError(401, "bad key"), openai_stream("recovered")],
    )
    patched_router(monkeypatch, transport)
    result = CliRunner().invoke(
        main,
        ["--model", "groq/x", "chat"],
        input="first\nsecond\n/exit\n",
        env={"EURUS_HOME": str(tmp_path)},
    )
    assert result.exit_code == 0
    assert "error:" in result.output
    assert "recovered" in result.output


def test_memory_cli_add_and_search(tmp_path):
    saved = run(
        tmp_path, "memory", "add", "approval protects shell actions", "--tag", "security shell"
    )
    assert saved.exit_code == 0
    assert "saved memory" in saved.output

    found = run(tmp_path, "memory", "search", "approval shell")
    assert found.exit_code == 0
    assert "approval protects shell actions" in found.output


def test_sessions_cli_lists_saved_session(tmp_path):
    from eurus.models import Message
    from eurus.sessions import SessionStore

    SessionStore(tmp_path / "sessions.sqlite3").save_messages(
        "cli-session", [Message.user("a saved question")]
    )
    result = run(tmp_path, "sessions", "list")
    assert result.exit_code == 0
    assert "cli-session" in result.output
    assert "a saved question" in result.output


def test_chat_persists_and_restores_named_session(tmp_path, monkeypatch):
    transport = FakeTransport(
        streams=[openai_stream("first answer"), openai_stream("second answer")]
    )
    patched_router(monkeypatch, transport)
    first = CliRunner().invoke(
        main,
        ["--model", "groq/x", "chat", "--session", "saved-chat"],
        input="first question\n/exit\n",
        env={"EURUS_HOME": str(tmp_path)},
    )
    second = CliRunner().invoke(
        main,
        ["--model", "groq/x", "chat", "--session", "saved-chat"],
        input="follow-up\n/exit\n",
        env={"EURUS_HOME": str(tmp_path)},
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert {"role": "user", "content": "first question"} in transport.calls[1].payload["messages"]
    assert {"role": "assistant", "content": "first answer"} in transport.calls[1].payload[
        "messages"
    ]


def test_metrics_command_reports_window_and_rates(tmp_path):
    store = cli_module.SessionStore(tmp_path / "sessions.sqlite3")
    store.record_event("approval_accepted")
    store.record_event("approval_rejected")
    result = run(tmp_path, "metrics", "--days", "7")
    assert result.exit_code == 0
    assert "last 7 days" in result.output
    assert "Approval decisions: 2" in result.output
    assert "Approval rejection rate: 50.0%" in result.output

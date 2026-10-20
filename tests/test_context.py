import pytest

from eurus.agent.context import ContextWindow, estimate_tokens
from eurus.config import load_config
from eurus.errors import ConfigError
from eurus.models import Message, ToolCall


def exchange(index, size):
    call = ToolCall(f"c{index}", "read_file", {"path": f"f{index}.py"})
    return [
        Message.assistant("", (call,)),
        Message.tool(f"c{index}", "x" * size, "read_file"),
    ]


def history(turns, size=4000):
    messages = [Message.system("sys")]
    for turn in range(turns):
        messages.append(Message.user(f"task {turn}"))
        messages.extend(exchange(turn, size))
        messages.append(Message.assistant(f"done {turn}"))
    return messages


def assert_valid(messages):
    seen = set()
    for message in messages:
        if message.role == "assistant":
            seen = {call.id for call in message.tool_calls}
        if message.role == "tool":
            assert message.tool_call_id in seen
    non_system = [m for m in messages if m.role != "system"]
    assert non_system[0].role == "user"


def test_small_history_is_unchanged():
    messages = history(2, size=40)
    assert ContextWindow(10_000).fit(messages) == messages


def test_old_tool_output_is_shrunk_before_dropping():
    messages = history(6)
    window = ContextWindow(4000, keep_recent=2, tool_output_chars=300)
    fitted = window.fit(messages)
    assert window.cost(fitted) <= 4000
    assert len(fitted) == len(messages)
    assert any("characters truncated" in m.content for m in fitted if m.role == "tool")
    assert_valid(fitted)


def test_old_turns_are_dropped_and_noted():
    messages = history(30)
    window = ContextWindow(3000, keep_recent=2, tool_output_chars=300)
    fitted = window.fit(messages)
    assert window.cost(fitted) <= 3000
    assert fitted[0].role == "system"
    assert "earlier messages were omitted" in fitted[0].content
    assert fitted[-1] == messages[-1]
    assert_valid(fitted)


def test_latest_user_message_survives_long_agent_run():
    messages = [Message.system("sys"), Message.user("old"), Message.assistant("ok")]
    messages.append(Message.user("current task"))
    for index in range(40):
        messages.extend(exchange(index, 3000))
    window = ContextWindow(2500, keep_recent=3, tool_output_chars=300)
    fitted = window.fit(messages)
    assert Message.user("current task") in fitted
    assert fitted[-1] == messages[-1]
    assert window.cost(fitted) <= 2500
    assert_valid(fitted)


def test_history_object_is_not_modified():
    messages = history(10)
    snapshot = list(messages)
    ContextWindow(2000).fit(messages)
    assert messages == snapshot


def test_estimate_counts_tool_arguments():
    plain = Message.assistant("")
    with_call = Message.assistant("", (ToolCall("1", "write", {"content": "y" * 400}),))
    assert estimate_tokens(with_call) > estimate_tokens(plain) + 90


def test_invalid_settings_rejected():
    with pytest.raises(ValueError):
        ContextWindow(0)


def test_context_tokens_config(tmp_path):
    env = {"EURUS_HOME": str(tmp_path)}
    assert load_config(cwd=tmp_path, environ=env).agent.context_tokens == 32000
    path = tmp_path / "eurus.toml"
    path.write_text("[agent]\ncontext_tokens = 10\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="context_tokens"):
        load_config(config_path=path, environ=env)


def test_agent_sends_fitted_history():
    import asyncio
    from pathlib import Path

    from eurus.agent import Agent
    from eurus.providers.testing import openai_stream, scripted_router
    from eurus.tools import ToolContext, default_registry

    router, transport = scripted_router(streams=[openai_stream("ok")])
    context = ToolContext(workspace=Path("."), max_output=1000, timeout=5.0)
    agent = Agent(router, default_registry(), context, system_prompt="sys", context_tokens=2000)
    agent.history.extend(history(30)[1:])

    async def go():
        return [event async for event in agent.run("now")]

    asyncio.run(go())
    sent = transport.calls[0].payload["messages"]
    assert len(sent) < len(agent.history)
    assert sent[-1]["content"] == "now"
    assert sent[0]["role"] == "system"


def test_context_compaction_keeps_rolling_summary_of_older_turns():
    messages = history(30, size=200)
    fitted = ContextWindow(1800, keep_recent=2, tool_output_chars=300).fit(messages)
    system = fitted[0].content
    assert "Earlier context summary" in system
    assert "User: task" in system
    assert fitted[-1] == messages[-1]


def test_context_window_accepts_custom_token_counter():
    class Counter:
        def count(self, message):
            return 17

    window = ContextWindow(100, token_counter=Counter())
    assert window.cost([Message.user("tiny")]) == 17

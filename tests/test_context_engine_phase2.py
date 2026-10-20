import asyncio
import sqlite3

from eurus.agent.context import ContextEngine, ExtractiveContextEngine, LLMSummarizingContextEngine
from eurus.models import Message, ToolCall
from eurus.sessions import SessionStore


def test_extractive_context_engine_keeps_recent_messages_and_tool_pairs():
    call = ToolCall("c1", "read_file", {"path": "a.py"})
    messages = [
        Message.user("old request"),
        Message.assistant("", (call,)),
        Message.tool("c1", "old result", "read_file"),
        Message.user("current request"),
        Message.assistant("current answer"),
    ]
    engine = ExtractiveContextEngine(keep_recent=2)
    result = asyncio.run(engine.compact(messages, max_tokens=10, token_counter=None))
    assert result.messages[-2:] == messages[-2:]
    assert any("Earlier context summary" in message.content for message in result.messages)
    assistant_calls = {call.id for message in result.messages for call in message.tool_calls}
    assert all(
        message.tool_call_id in assistant_calls
        for message in result.messages
        if message.role == "tool"
    )


def test_llm_summarizer_falls_back_to_extractive_when_provider_fails():
    class BrokenRouter:
        async def chat(self, *args, **kwargs):
            raise RuntimeError("offline")

    messages = [Message.user("old"), Message.assistant("answer"), Message.user("current")]
    engine = LLMSummarizingContextEngine(BrokenRouter(), keep_recent=1)
    result = asyncio.run(engine.compact(messages, max_tokens=10, token_counter=None))
    assert result.used_fallback
    assert result.summary
    assert result.messages[-1] == messages[-1]


def test_session_lineage_migration_is_additive(tmp_path):
    path = tmp_path / "sessions.sqlite3"
    store = SessionStore(path)
    store.save_messages("parent", [Message.user("hello")])
    store.save_session_lineage("parent", "child", "context_compaction", "summary")
    assert store.session_lineage("child") == {
        "parent_id": "parent",
        "child_id": "child",
        "reason": "context_compaction",
        "summary": "summary",
    }
    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {"sessions", "messages", "session_summaries", "session_lineage"}.issubset(tables)


def test_context_engine_protocol_is_runtime_checkable_by_shape():
    assert hasattr(ContextEngine, "compact")


def test_agent_context_preflight_flushes_and_records_lineage(tmp_path):
    from eurus.agent.loop import Agent
    from eurus.providers.testing import scripted_router
    from eurus.tools import ToolContext, default_registry

    router, _ = scripted_router()
    store = SessionStore(tmp_path / "sessions.sqlite3")
    store.save_messages("parent", [Message.user("previous")])
    calls = []

    class Engine(ExtractiveContextEngine):
        async def compact(self, messages, *, max_tokens, token_counter=None):
            calls.append("compact")
            return await super().compact(
                messages, max_tokens=max_tokens, token_counter=token_counter
            )

    agent = Agent(
        router,
        default_registry(),
        ToolContext(workspace=tmp_path, max_output=1000, timeout=5),
        system_prompt="system",
        context_tokens=2000,
        context_engine=Engine(keep_recent=2),
        context_store=store,
        session_id="parent",
        flush_memory=lambda: calls.append("flush"),
    )
    agent.history.extend([Message.user("old " * 1000), Message.assistant("answer " * 1000)])
    asyncio.run(agent._preflight_context())
    assert calls == ["flush", "compact"]
    assert agent.session_id != "parent"
    assert store.session_lineage(agent.session_id)["parent_id"] == "parent"


def test_gateway_context_preflight_uses_higher_threshold(tmp_path):
    from eurus.agent.loop import Agent
    from eurus.providers.testing import scripted_router
    from eurus.tools import ToolContext, default_registry

    router, _ = scripted_router()
    agent = Agent(
        router,
        default_registry(),
        ToolContext(workspace=tmp_path, max_output=1000, timeout=5),
        system_prompt="system",
        context_tokens=2000,
        context_mode="gateway",
    )
    agent.history.append(Message.user("x " * 700))
    asyncio.run(agent._preflight_context())
    assert not any(
        message.content.startswith("Earlier context summary:") for message in agent.history
    )

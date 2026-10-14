from __future__ import annotations

from axiom.models import Message, ToolCall
from axiom.sessions import SessionStore


def test_session_messages_round_trip_and_system_prompt_is_not_saved(tmp_path):
    store = SessionStore(tmp_path / "sessions.sqlite3")
    messages = [
        Message.system("private runtime prompt"),
        Message.user("hello"),
        Message.assistant("working", (ToolCall("call-1", "calculate", {"expression": "2+2"}),)),
        Message.tool("call-1", "4", "calculate"),
        Message.assistant("4"),
    ]

    store.save_messages("test-session", messages)

    assert store.load_messages("test-session") == messages[1:]
    sessions = store.list_sessions()
    assert sessions[0]["id"] == "test-session"
    assert sessions[0]["title"] == "hello"
    assert sessions[0]["message_count"] == 4


def test_delete_session_is_cascading_and_reports_missing(tmp_path):
    store = SessionStore(tmp_path / "sessions.sqlite3")
    store.save_messages("to-delete", [Message.user("hello")])

    assert store.delete_session("to-delete") is True
    assert store.load_messages("to-delete") == []
    assert store.delete_session("to-delete") is False


def test_memory_fts_search_tags_and_delete_safety(tmp_path):
    store = SessionStore(tmp_path / "sessions.sqlite3")
    store.add_memory("AXIOM should require approval for shell mutations", "security shell")
    store.add_memory("SQLite stores persistent chat sessions", "storage")

    results = store.search_memories("approval shell")

    assert len(results) == 1
    assert "require approval" in results[0]["content"]
    assert results[0]["tags"] == "security shell"
    assert store.search_memories('" OR *') == []


def test_memory_validation(tmp_path):
    store = SessionStore(tmp_path / "sessions.sqlite3")
    try:
        store.add_memory("  ")
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("empty memory should be rejected")

    try:
        store.search_memories("term", limit=101)
    except ValueError as exc:
        assert "between 1 and 100" in str(exc)
    else:
        raise AssertionError("invalid limit should be rejected")


def test_database_permissions_are_private_on_posix(tmp_path):
    import os
    import stat

    if os.name != "posix":
        return
    path = tmp_path / "sessions.sqlite3"
    SessionStore(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

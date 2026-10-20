from __future__ import annotations

from eurus.models import Message, ToolCall
from eurus.sessions import SessionStore


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
    store.add_memory("EURUS should require approval for shell mutations", "security shell")
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


def test_typed_memory_metadata_and_usage_tracking(tmp_path):
    store = SessionStore(tmp_path / "sessions.sqlite3")
    memory_id = store.add_memory(
        "Prefer concise Python comments",
        category="user",
        memory_type="preference",
        source="explicit_correction",
        confidence=0.9,
    )
    rows = store.relevant_memories(["concise", "Python"], limit=5)
    assert rows[0]["id"] == memory_id
    assert rows[0]["memory_type"] == "preference"
    assert rows[0]["source"] == "explicit_correction"
    assert rows[0]["confidence"] == 0.9
    assert store.list_memories()[0]["use_count"] == 1


def test_typed_memory_rejects_invalid_metadata(tmp_path):
    store = SessionStore(tmp_path / "sessions.sqlite3")
    try:
        store.add_memory("test", memory_type="system_override")
    except ValueError as exc:
        assert "unsupported memory type" in str(exc)
    else:
        raise AssertionError("unsupported memory type should be rejected")


def test_session_summary_round_trip(tmp_path):
    store = SessionStore(tmp_path / "sessions.sqlite3")
    store.save_messages("summary-session", [Message.user("build the parser")])
    store.save_session_summary("summary-session", "Parser design and outstanding tests")
    assert store.load_session_summary("summary-session") == "Parser design and outstanding tests"


def test_memory_consolidation_is_dry_run_by_default(tmp_path):
    store = SessionStore(tmp_path / "sessions.sqlite3")
    store.add_memory("Use concise comments", category="style", memory_type="preference")
    store.add_memory("Use concise comments", category="style", memory_type="preference")
    planned = store.consolidate_memories()
    assert planned == {"scanned": 2, "duplicates": 1, "deleted": 0}
    assert len(store.list_memories()) == 2
    applied = store.consolidate_memories(dry_run=False)
    assert applied["deleted"] == 1
    assert len(store.list_memories()) == 1


def test_lifecycle_event_metrics_count_approval_and_correction_events(tmp_path):
    store = SessionStore(tmp_path / "sessions.sqlite3")
    store.record_event("approval_rejected", session_id="s1", metadata={"tool": "shell"})
    store.record_event("correction_proposed", session_id="s1")
    store.record_event("correction_accepted", session_id="s1")

    assert store.event_metrics() == {
        "approval_rejected": 1,
        "correction_accepted": 1,
        "correction_proposed": 1,
    }


def test_lifecycle_event_validation_and_metadata_limit(tmp_path):
    store = SessionStore(tmp_path / "sessions.sqlite3")
    try:
        store.record_event("Bad Event")
    except ValueError as exc:
        assert "invalid event type" in str(exc)
    else:
        raise AssertionError("invalid event type should fail")

    try:
        store.record_event("test_event", metadata={"value": "x" * 2100})
    except ValueError as exc:
        assert "metadata exceeds" in str(exc)
    else:
        raise AssertionError("oversized event metadata should fail")


def test_build_session_summary_is_bounded_and_uses_saved_messages(tmp_path):
    store = SessionStore(tmp_path / "sessions.sqlite3")
    store.save_messages(
        "summary-build",
        [Message.user("  Build   a parser  "), Message.assistant("Parser tests are passing.")],
    )

    summary = store.build_session_summary("summary-build", max_entries=2, max_chars=500)
    assert "User: Build a parser" in summary
    assert "Agent: Parser tests are passing." in summary
    assert len(summary) <= 500


def test_event_metrics_window_and_rates(tmp_path):
    store = SessionStore(tmp_path / "sessions.sqlite3")
    store.record_event("approval_accepted")
    store.record_event("approval_rejected")
    store.record_event("correction_proposed")
    store.record_event("correction_accepted")
    store.record_event("correction_rejected")

    assert store.event_metrics(days=30) == {
        "approval_accepted": 1,
        "approval_rejected": 1,
        "correction_accepted": 1,
        "correction_proposed": 1,
        "correction_rejected": 1,
    }
    assert store.event_rates(days=30) == {
        "days": 30,
        "approval_decisions": 2,
        "approval_rejection_rate": 0.5,
        "correction_proposals": 1,
        "correction_decisions": 2,
        "correction_acceptance_rate": 0.5,
        "correction_rejection_rate": 0.5,
    }


def test_event_metrics_rejects_invalid_window(tmp_path):
    store = SessionStore(tmp_path / "sessions.sqlite3")
    try:
        store.event_metrics(days=0)
    except ValueError as exc:
        assert "between 1 and 3650" in str(exc)
    else:
        raise AssertionError("invalid metrics window should fail")

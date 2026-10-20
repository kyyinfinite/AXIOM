from pathlib import Path

from eurus.memory.manager import (
    MemoryManager,
    MemoryProvider,
    SQLiteMemoryProvider,
    build_memory_provider,
)
from eurus.models import Message
from eurus.sessions import SessionStore


def test_session_search_indexes_messages_and_respects_project_scope(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.sqlite3")
    store.save_messages(
        "project-a", [Message.user("decision use sqlite fts search")], project_scope="/project/a"
    )
    store.save_messages(
        "project-b", [Message.user("decision use sqlite fts search")], project_scope="/project/b"
    )

    rows = store.search_session_messages("sqlite search", project_scope="/project/a")

    assert [row["session_id"] for row in rows] == ["project-a"]
    assert rows[0]["role"] == "user"


def test_session_search_retrieves_historical_messages(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.sqlite3")
    store.save_messages(
        "history",
        [
            Message.user("Remember the migration decision"),
            Message.assistant("Use additive SQLite migrations"),
        ],
    )

    rows = store.search_session_messages("additive migrations")

    assert len(rows) == 1
    assert rows[0]["content"] == "Use additive SQLite migrations"


def test_memory_search_is_scoped_and_global_memories_are_shared(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.sqlite3")
    store.add_memory("Use SQLite for project persistence", scope="/project/a")
    store.add_memory("Use SQLite for global preferences", scope="global")

    rows = store.search_memories("SQLite", scope="/project/a")

    assert {row["scope"] for row in rows} == {"/project/a", "global"}
    assert len(store.search_memories("SQLite", scope="/project/b")) == 1


def test_memory_provider_is_single_select_and_sqlite_backed(tmp_path: Path) -> None:
    provider = build_memory_provider("sqlite", tmp_path / "sessions.sqlite3")

    assert isinstance(provider, MemoryProvider)
    assert isinstance(provider, SQLiteMemoryProvider)
    memory_id = provider.remember("Keep concise answers", category="preference")
    assert provider.recall("concise answers")[0]["id"] == memory_id
    assert provider.forget(memory_id)


def test_memory_provider_rejects_unknown_backend(tmp_path: Path) -> None:
    try:
        build_memory_provider("unknown", tmp_path / "sessions.sqlite3")
    except ValueError as exc:
        assert "unsupported memory provider" in str(exc)
    else:
        raise AssertionError("unknown provider should fail closed")


def test_curated_context_is_bounded_and_scope_aware(tmp_path: Path) -> None:
    manager = MemoryManager(tmp_path / "sessions.sqlite3")
    manager.store.add_memory(
        "Prefer direct concise answers", memory_type="preference", scope="global"
    )
    manager.store.add_memory("Project uses Python", memory_type="project_fact", scope="/project/a")
    manager.store.add_memory(
        "Private other project detail", memory_type="project_fact", scope="/project/b"
    )

    context = manager.curated_context(scope="/project/a", max_chars=180)

    assert any("concise answers" in item for item in context)
    assert any("Project uses Python" in item for item in context)
    assert all("Private other project" not in item for item in context)
    assert sum(map(len, context)) <= 180


def test_privacy_export_and_purge_cover_saved_personal_data(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.sqlite3")
    store.save_messages("private-session", [Message.user("private prompt")])
    store.add_memory("private note")
    store.record_event("test_event", session_id="private-session")

    exported = store.export_privacy_data()
    counts = store.purge_privacy_data()

    assert len(exported["sessions"]) == 1
    assert len(exported["messages"]) == 1
    assert len(exported["memories"]) == 1
    assert counts == {"sessions": 1, "messages": 1, "memories": 1, "events": 1}
    assert store.export_privacy_data() == {
        "sessions": [],
        "messages": [],
        "memories": [],
        "session_summaries": [],
        "session_lineage": [],
    }


def test_compacted_session_can_resume_from_original_id(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.sqlite3")
    store.save_messages("original", [Message.user("first")])
    store.save_session_lineage("original", "compacted", "compact", "earlier context")
    store.save_messages("compacted", [Message.user("recent")])

    assert store.latest_session_descendant("original") == "compacted"


def test_memory_curation_nudge_is_rate_limited_by_event(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.sqlite3")

    assert store.memory_curation_due()
    store.record_event("memory_curation_nudge")
    assert not store.memory_curation_due()


def test_existing_session_database_is_backfilled_into_fts(tmp_path: Path) -> None:
    import json
    import sqlite3

    database = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        "CREATE TABLE sessions(id TEXT PRIMARY KEY, title TEXT NOT NULL, "
        "created_at TEXT NOT NULL, updated_at TEXT NOT NULL);"
        "CREATE TABLE messages(session_id TEXT NOT NULL REFERENCES sessions(id) "
        "ON DELETE CASCADE, sequence INTEGER NOT NULL, payload TEXT NOT NULL, "
        "created_at TEXT NOT NULL, PRIMARY KEY(session_id, sequence));"
    )
    connection.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?)",
        ("legacy", "legacy", "2026-01-01", "2026-01-01"),
    )
    connection.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?)",
        (
            "legacy",
            0,
            json.dumps({"role": "user", "content": "backfill historic message search"}),
            "2026-01-01",
        ),
    )
    connection.commit()
    connection.close()

    store = SessionStore(database)

    rows = store.search_session_messages("historic message")
    assert len(rows) == 1
    assert rows[0]["session_id"] == "legacy"

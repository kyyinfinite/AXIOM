from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from eurus.agent.planner import TaskPlanner
from eurus.coding.tools import coding_tools
from eurus.memory.manager import MemoryManager
from eurus.models import ToolCall
from eurus.sessions import SessionStore
from eurus.tools import ToolContext, ToolRegistry, default_registry


def run_tool(registry: ToolRegistry, name: str, arguments: dict, workspace: Path):
    return asyncio.run(
        registry.execute(
            ToolCall("test-call", name, arguments),
            ToolContext(workspace=workspace, max_output=20_000, timeout=5),
        )
    )


def test_default_registry_stays_compatible_and_capability_tools_are_opt_in():
    assert default_registry().names() == ["calculate", "get_time"]
    names = default_registry(include_capability_tools=True).names()
    assert "apply_patch" in names
    assert "recall_memory" in names
    assert "plan_task" in names


def test_memory_manager_categories_update_recall_and_delete(tmp_path):
    manager = MemoryManager(tmp_path / "memory.sqlite3")
    memory_id = manager.remember(
        "EURUS project must preserve SQLite session compatibility",
        category="project",
        tags="eurus sqlite compatibility",
    )
    results = manager.recall("SQLite compatibility")
    assert results[0]["id"] == memory_id
    assert results[0]["category"] == "project"
    assert manager.update(memory_id, "EURUS project uses Python tests", category="decision")
    assert manager.recall("Python tests")[0]["category"] == "decision"
    assert manager.forget(memory_id)
    assert not manager.forget(memory_id)


def test_legacy_original_memory_schema_migrates_additively(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE memories_fts USING fts5(
            content, tags, content='memories', content_rowid='id'
        );
        CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content, tags)
            VALUES (new.id, new.content, new.tags);
        END;
        CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, tags)
            VALUES ('delete', old.id, old.content, old.tags);
        END;
        CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, tags)
            VALUES ('delete', old.id, old.content, old.tags);
            INSERT INTO memories_fts(rowid, content, tags)
            VALUES (new.id, new.content, new.tags);
        END;
        """
    )
    connection.execute(
        "INSERT INTO memories(content, tags, created_at) VALUES (?, ?, ?)",
        ("legacy session memory", "legacy", "2026-01-01T00:00:00+00:00"),
    )
    connection.commit()
    connection.close()

    store = SessionStore(path)
    result = store.search_memories("legacy session")
    assert result[0]["category"] == "general"
    assert result[0]["updated_at"] == "2026-01-01T00:00:00+00:00"


def test_planner_is_bounded_and_rejects_empty_goals():
    plan = TaskPlanner(max_steps=3).create_plan("fix a bug and test it")
    assert len(plan.steps) == 3
    assert plan.steps[0].title == "Inspect workspace"
    with pytest.raises(ValueError):
        TaskPlanner().create_plan("  ")
    with pytest.raises(ValueError):
        TaskPlanner(max_steps=9)


def test_coding_tools_read_search_and_propose_without_mutating(tmp_path):
    (tmp_path / "app.py").write_text("def hello():\n    return 'hi'\n", encoding="utf-8")
    registry = ToolRegistry(coding_tools())

    read = run_tool(registry, "read_file", {"path": "app.py"}, tmp_path)
    assert "def hello" in read.output
    search = run_tool(registry, "search_code", {"query": "hello"}, tmp_path)
    assert "app.py:1" in search.output
    proposal = run_tool(
        registry,
        "propose_patch",
        {"path": "app.py", "new_content": "def hello():\n    return 'hello'\n"},
        tmp_path,
    )
    assert "-    return 'hi'" in proposal.output
    assert (tmp_path / "app.py").read_text(encoding="utf-8").endswith("'hi'\n")


def test_coding_tools_reject_path_escape_and_stale_patch(tmp_path):
    (tmp_path / "app.py").write_text("safe\n", encoding="utf-8")
    registry = ToolRegistry(coding_tools())
    escaped = run_tool(registry, "read_file", {"path": "../outside.txt"}, tmp_path)
    assert escaped.is_error
    assert "outside" in escaped.output
    stale = run_tool(
        registry,
        "apply_patch",
        {"path": "app.py", "expected_content": "old\n", "new_content": "changed\n"},
        tmp_path,
    )
    assert stale.is_error
    assert "changed since it was read" in stale.output
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "safe\n"


def test_apply_patch_is_marked_as_approval_required(tmp_path):
    tool = ToolRegistry(coding_tools()).get("apply_patch")
    assert tool is not None
    assert tool.mutates
    assert tool.requires_approval


def test_coding_tools_reject_symbolic_links(tmp_path):
    target = tmp_path / "real.txt"
    target.write_text("secret", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links unavailable on this platform")
    registry = ToolRegistry(coding_tools())
    result = run_tool(registry, "read_file", {"path": "link.txt"}, tmp_path)
    assert result.is_error
    assert "symbolic links" in result.output


def test_apply_patch_changes_only_when_expected_content_matches(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("before\n", encoding="utf-8")
    registry = ToolRegistry(coding_tools())
    result = run_tool(
        registry,
        "apply_patch",
        {"path": "module.py", "expected_content": "before\n", "new_content": "after\n"},
        tmp_path,
    )
    assert not result.is_error
    assert path.read_text(encoding="utf-8") == "after\n"

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from eurus.coding.changes import WorkspaceChangeManager
from eurus.coding.patchset import (
    FilePatch,
    PatchError,
    apply_patchset,
    preview_patchset,
)
from eurus.coding.project_index import index_project
from eurus.coding.validation import run_validation, validate_command


def test_project_index_finds_python_symbols_and_skips_venv(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("import json\n\ndef run():\n    return json.dumps({})\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "hidden.py").write_text("class Hidden: pass\n")

    results = index_project(tmp_path)

    assert [item.path for item in results] == ["app.py"]
    assert results[0].imports == ("json",)
    assert "run" in results[0].definitions


def test_patchset_updates_multiple_files_and_restores_only_owned_changes(tmp_path: Path) -> None:
    (tmp_path / "one.py").write_text("VALUE = 1\n")
    patches = [
        FilePatch("one.py", "VALUE = 1\n", "VALUE = 2\n"),
        FilePatch("new.py", None, "NEW = True\n"),
    ]
    assert "one.py" in preview_patchset(patches)
    result = apply_patchset(tmp_path, patches)
    assert (tmp_path / "one.py").read_text() == "VALUE = 2\n"
    assert (tmp_path / "new.py").read_text() == "NEW = True\n"

    manager = WorkspaceChangeManager(tmp_path)
    manager.record(result.changes)
    assert manager.review().changed_files == ("one.py", "new.py")
    assert manager.restore() == ["new.py", "one.py"]
    assert (tmp_path / "one.py").read_text() == "VALUE = 1\n"
    assert not (tmp_path / "new.py").exists()


def test_patchset_rejects_stale_file_without_partial_write(tmp_path: Path) -> None:
    (tmp_path / "one.py").write_text("CHANGED = True\n")
    (tmp_path / "two.py").write_text("VALUE = 2\n")
    patches = [
        FilePatch("two.py", "VALUE = 2\n", "VALUE = 3\n"),
        FilePatch("one.py", "OLD = True\n", "NEW = True\n"),
    ]
    with pytest.raises(PatchError, match="changed since planning"):
        apply_patchset(tmp_path, patches)
    assert (tmp_path / "two.py").read_text() == "VALUE = 2\n"


def test_restore_refuses_to_overwrite_later_user_change(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("A = 1\n")
    result = apply_patchset(tmp_path, [FilePatch("app.py", "A = 1\n", "A = 2\n")])
    (tmp_path / "app.py").write_text("USER_CHANGE = True\n")
    manager = WorkspaceChangeManager(tmp_path)
    manager.record(result.changes)
    assert not manager.review().can_restore
    with pytest.raises(PatchError, match="later change"):
        manager.restore()
    assert (tmp_path / "app.py").read_text() == "USER_CHANGE = True\n"


def test_validation_command_is_allowlisted_and_shell_free(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        validate_command(["sh", "-c", "rm -rf ."])
    result = run_validation(tmp_path, [sys.executable, "-m", "compileall", "-q", "."])
    assert result.passed


def test_workflow_tools_are_opt_in_and_multi_file_apply_requires_approval() -> None:
    from eurus.coding.workflow_tools import workflow_tools
    from eurus.tools.builtin import default_registry

    legacy = default_registry()
    assert "index_project" not in legacy
    tools = {tool.name: tool for tool in workflow_tools()}
    assert tools["apply_patchset"].mutates
    assert tools["apply_patchset"].requires_approval
    expanded = default_registry(include_capability_tools=True, include_workflow_tools=True)
    assert {"index_project", "preview_patchset", "apply_patchset"}.issubset(set(expanded.names()))


def test_change_manager_tracks_sequential_edits_to_same_file(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\n")
    first = apply_patchset(tmp_path, [FilePatch("app.py", "VALUE = 1\n", "VALUE = 2\n")])
    second = apply_patchset(tmp_path, [FilePatch("app.py", "VALUE = 2\n", "VALUE = 3\n")])
    manager = WorkspaceChangeManager(tmp_path)
    manager.record(first.changes)
    manager.record(second.changes)
    assert manager.review().changed_files == ("app.py",)
    assert manager.restore() == ["app.py"]
    assert (tmp_path / "app.py").read_text() == "VALUE = 1\n"

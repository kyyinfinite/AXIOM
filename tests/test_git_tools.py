import asyncio
import subprocess
from pathlib import Path

from eurus.coding.git_tools import git_tools
from eurus.models import ToolCall
from eurus.tools.base import ToolContext, ToolRegistry


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)


def _execute(registry: ToolRegistry, name: str, arguments: dict, workspace: Path):
    return asyncio.run(
        registry.execute(ToolCall("test", name, arguments), ToolContext(workspace=workspace))
    )


def _repo(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    (path / "sample.txt").write_text("first\n", encoding="utf-8")
    _git(path, "add", "sample.txt")
    _git(path, "commit", "-m", "initial", "-q")
    return path


def test_git_tools_read_repository_state(tmp_path):
    workspace = _repo(tmp_path / "repo")
    registry = ToolRegistry(git_tools())

    status = _execute(registry, "git_status", {}, workspace)
    log = _execute(registry, "git_log", {"limit": 5}, workspace)
    branches = _execute(registry, "git_branches", {}, workspace)

    assert not status.is_error
    assert "sample.txt" not in status.output
    assert "initial" in log.output
    assert "master" in branches.output or "main" in branches.output


def test_git_diff_reads_unstaged_and_staged_changes(tmp_path):
    workspace = _repo(tmp_path / "repo")
    registry = ToolRegistry(git_tools())
    (workspace / "sample.txt").write_text("second\n", encoding="utf-8")

    result = _execute(registry, "git_diff", {}, workspace)
    assert not result.is_error
    assert "second" in result.output

    _git(workspace, "add", "sample.txt")
    staged = _execute(registry, "git_diff", {"staged": True}, workspace)
    assert not staged.is_error
    assert "second" in staged.output


def test_git_tools_reject_non_repository_and_invalid_revision(tmp_path):
    registry = ToolRegistry(git_tools())
    not_repo = tmp_path / "plain"
    not_repo.mkdir()
    result = _execute(registry, "git_status", {}, not_repo)
    assert result.is_error
    assert "not a Git repository" in result.output

    workspace = _repo(tmp_path / "repo")
    invalid = _execute(registry, "git_show", {"revision": "--help"}, workspace)
    assert invalid.is_error
    assert "commit hash" in invalid.output


def test_git_show_accepts_commit_hash(tmp_path):
    workspace = _repo(tmp_path / "repo")
    revision = subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = _execute(ToolRegistry(git_tools()), "git_show", {"revision": revision}, workspace)
    assert not result.is_error
    assert "initial" in result.output

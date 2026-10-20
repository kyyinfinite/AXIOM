from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from eurus.errors import ToolError
from eurus.tools.base import Tool, ToolContext

REVISION_PATTERN = re.compile(r"^[0-9a-fA-F]{7,40}$")
GIT_TIMEOUT = 8.0


async def _run_git(context: ToolContext, *args: str) -> str:
    root = context.workspace.resolve()
    if not root.is_dir() or not ((root / ".git").exists()):
        raise ToolError("workspace is not a Git repository")
    env = os.environ.copy()
    env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.fsmonitor",
            "GIT_CONFIG_VALUE_0": "false",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    try:
        process = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(root),
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=GIT_TIMEOUT)
    except FileNotFoundError as exc:
        raise ToolError("Git executable is not installed") from exc
    except TimeoutError as exc:
        process.kill()
        await process.communicate()
        raise ToolError("Git command timed out") from exc
    output = stdout.decode("utf-8", errors="replace").strip()
    error = stderr.decode("utf-8", errors="replace").strip()
    if process.returncode:
        raise ToolError(error[:500] or f"Git exited with status {process.returncode}")
    return output or "(no output)"


async def _status(arguments: dict[str, Any], context: ToolContext) -> str:
    return await _run_git(context, "status", "--short", "--branch", "--untracked-files=normal")


async def _diff(arguments: dict[str, Any], context: ToolContext) -> str:
    staged = bool(arguments.get("staged", False))
    args = ["diff", "--no-ext-diff", "--no-color", "--"]
    if staged:
        args.insert(1, "--cached")
    return await _run_git(context, *args)


async def _log(arguments: dict[str, Any], context: ToolContext) -> str:
    limit = int(arguments.get("limit", 10))
    if not 1 <= limit <= 30:
        raise ToolError("limit must be between 1 and 30")
    return await _run_git(context, "log", f"-{limit}", "--oneline", "--decorate=no")


async def _branches(arguments: dict[str, Any], context: ToolContext) -> str:
    return await _run_git(context, "branch", "--list", "--no-color")


async def _show(arguments: dict[str, Any], context: ToolContext) -> str:
    revision = str(arguments["revision"]).strip()
    if not REVISION_PATTERN.fullmatch(revision):
        raise ToolError("revision must be a 7- to 40-character commit hash")
    return await _run_git(context, "show", "--stat", "--oneline", "--no-renames", revision, "--")


def git_tools() -> list[Tool]:
    return [
        Tool(
            "git_status",
            "Show branch and working-tree status for the configured Git workspace.",
            {"type": "object", "properties": {}, "additionalProperties": False},
            _status,
        ),
        Tool(
            "git_diff",
            "Show unstaged or staged changes without running external diff commands.",
            {
                "type": "object",
                "properties": {"staged": {"type": "boolean"}},
                "additionalProperties": False,
            },
            _diff,
        ),
        Tool(
            "git_log",
            "Show recent commits, limited to 30 entries.",
            {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
                "additionalProperties": False,
            },
            _log,
        ),
        Tool(
            "git_branches",
            "List local branches without changing the current branch.",
            {"type": "object", "properties": {}, "additionalProperties": False},
            _branches,
        ),
        Tool(
            "git_show",
            "Show summary statistics for a commit identified by its full or abbreviated hash.",
            {
                "type": "object",
                "properties": {"revision": {"type": "string"}},
                "required": ["revision"],
                "additionalProperties": False,
            },
            _show,
        ),
    ]

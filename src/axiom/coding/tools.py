from __future__ import annotations

import difflib
import hashlib
from pathlib import Path
from typing import Any

from axiom.errors import ToolError
from axiom.tools.base import Tool, ToolContext

MAX_FILE_BYTES = 1_000_000
MIN_HASH_PREFIX = 12
MAX_SEARCH_RESULTS = 100
IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache"}


def _resolve(context: ToolContext, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ToolError("path must be relative to the workspace")
    root = context.workspace.resolve()
    raw_target = root / candidate
    cursor = root
    for part in candidate.parts:
        if part in {"", "."}:
            continue
        cursor = cursor / part
        if cursor.is_symlink():
            raise ToolError("symbolic links are not allowed in coding tool paths")
    target = raw_target.resolve()
    if target != root and root not in target.parents:
        raise ToolError("path is outside the configured workspace")
    return target


def _read_text(path: Path) -> str:
    try:
        if not path.is_file():
            raise ToolError("file does not exist or is not a regular file")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ToolError("file exceeds 1000000 bytes")
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError("file is not valid UTF-8 text") from exc
    except OSError as exc:
        raise ToolError(f"cannot read file: {exc}") from exc


async def _list_files(arguments: dict[str, Any], context: ToolContext) -> str:
    relative = str(arguments.get("path", "."))
    target = _resolve(context, relative)
    if not target.is_dir():
        raise ToolError("path is not a directory")
    limit = int(arguments.get("limit", 100))
    if not 1 <= limit <= 500:
        raise ToolError("limit must be between 1 and 500")
    found: list[str] = []
    for item in sorted(target.rglob("*")):
        try:
            if any(
                part in IGNORED_DIRS for part in item.relative_to(context.workspace.resolve()).parts
            ):
                continue
            resolved = item.resolve()
            root = context.workspace.resolve()
            if resolved != root and root not in resolved.parents:
                continue
            if item.is_file() and not item.is_symlink():
                found.append(item.relative_to(root).as_posix())
                if len(found) >= limit:
                    break
        except (OSError, ValueError):
            continue
    return "\n".join(found) if found else "(no files found)"


async def _read_file(arguments: dict[str, Any], context: ToolContext) -> str:
    target = _resolve(context, str(arguments["path"]))
    if target.is_symlink():
        raise ToolError("symbolic links are not readable through coding tools")
    return _read_text(target)


async def _search_code(arguments: dict[str, Any], context: ToolContext) -> str:
    query = str(arguments["query"])
    if not query or len(query) > 500:
        raise ToolError("query must be 1 to 500 characters")
    relative = str(arguments.get("path", "."))
    start = _resolve(context, relative)
    if not start.exists():
        raise ToolError("search path does not exist")
    limit = int(arguments.get("limit", 30))
    if not 1 <= limit <= MAX_SEARCH_RESULTS:
        raise ToolError("limit must be between 1 and 100")
    files = [start] if start.is_file() else sorted(start.rglob("*"))
    root = context.workspace.resolve()
    matches: list[str] = []
    for file in files:
        if len(matches) >= limit:
            break
        try:
            relative_parts = file.relative_to(root).parts
            if any(part in IGNORED_DIRS for part in relative_parts):
                continue
            if file.is_symlink() or not file.is_file() or file.stat().st_size > MAX_FILE_BYTES:
                continue
            resolved = file.resolve()
            if resolved != root and root not in resolved.parents:
                continue
            text = file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            if query.casefold() in line.casefold():
                matches.append(f"{file.relative_to(root).as_posix()}:{line_number}: {line[:500]}")
                if len(matches) >= limit:
                    break
    return "\n".join(matches) if matches else "(no matches)"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def _propose_patch(arguments: dict[str, Any], context: ToolContext) -> str:
    relative = str(arguments["path"])
    target = _resolve(context, relative)
    if target.is_symlink():
        raise ToolError("symbolic links cannot be patched")
    old = _read_text(target)
    new = str(arguments["new_content"])
    if len(new.encode("utf-8")) > MAX_FILE_BYTES:
        raise ToolError("proposed file exceeds 1000000 bytes")
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{relative}",
        tofile=f"b/{relative}",
    )
    output = "".join(diff)
    if not output:
        return "No changes proposed."
    return f"base_sha256: {_digest(old)}\n{output}"


async def _apply_patch(arguments: dict[str, Any], context: ToolContext) -> str:
    relative = str(arguments["path"])
    target = _resolve(context, relative)
    if target.is_symlink():
        raise ToolError("symbolic links cannot be patched")
    if not target.exists() or not target.is_file():
        raise ToolError("target file does not exist")
    replacement = str(arguments["new_content"])
    current = _read_text(target)
    base = arguments.get("base_sha256")
    expected = arguments.get("expected_content")
    if base is not None:
        prefix = str(base).strip().lower()
        if len(prefix) < MIN_HASH_PREFIX or not _digest(current).startswith(prefix):
            raise ToolError(
                "base_sha256 does not match the current file; run propose_patch again "
                "and use the base_sha256 it returns"
            )
    elif expected is None:
        raise ToolError("provide base_sha256 from propose_patch, or expected_content")
    elif current != str(expected):
        if str(expected) == replacement:
            raise ToolError(
                "expected_content must be the file's current content before your change, "
                "but it equals new_content. Run propose_patch and pass the base_sha256 it "
                "returns instead"
            )
        raise ToolError("file changed since it was read; re-read and propose a fresh patch")
    if len(replacement.encode("utf-8")) > MAX_FILE_BYTES:
        raise ToolError("replacement exceeds 1000000 bytes")
    try:
        target.write_text(replacement, encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"cannot write file: {exc}") from exc
    relative_path = target.relative_to(context.workspace.resolve()).as_posix()
    return f"updated {relative_path} ({len(current)} -> {len(replacement)} characters)"


def coding_tools() -> list[Tool]:
    string = {"type": "string"}
    return [
        Tool(
            "list_files",
            "List files inside the configured workspace, skipping common generated directories.",
            {
                "type": "object",
                "properties": {"path": string, "limit": {"type": "integer"}},
                "additionalProperties": False,
            },
            _list_files,
        ),
        Tool(
            "read_file",
            "Read a UTF-8 text file inside the workspace. Files over 1 MB and symbolic links "
            "are rejected.",
            {
                "type": "object",
                "properties": {"path": string},
                "required": ["path"],
                "additionalProperties": False,
            },
            _read_file,
        ),
        Tool(
            "search_code",
            "Search text in workspace files and return file/line matches.",
            {
                "type": "object",
                "properties": {
                    "query": string,
                    "path": string,
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            _search_code,
        ),
        Tool(
            "propose_patch",
            "Show a unified diff for proposed full-file content without modifying the file.",
            {
                "type": "object",
                "properties": {"path": string, "new_content": string},
                "required": ["path", "new_content"],
                "additionalProperties": False,
            },
            _propose_patch,
        ),
        Tool(
            "apply_patch",
            "Replace a file with new_content. Pass base_sha256 from propose_patch (preferred), "
            "or expected_content, which must be the file's current content before the change. "
            "Requires owner approval.",
            {
                "type": "object",
                "properties": {
                    "path": string,
                    "new_content": string,
                    "base_sha256": string,
                    "expected_content": string,
                },
                "required": ["path", "new_content"],
                "additionalProperties": False,
            },
            _apply_patch,
            mutates=True,
            requires_approval=True,
        ),
    ]


def describe_diff(old: str, new: str) -> str:
    """Utility kept public for tests and integrations that want a unified diff."""
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True)))

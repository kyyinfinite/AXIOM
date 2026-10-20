from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from eurus.coding.document_tools import (
    _document_create,
    _document_read,
    _document_replace,
    document_tools,
)
from eurus.errors import ToolError
from eurus.tools.base import ToolContext


def context(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path)


def test_document_tools_are_exposed() -> None:
    assert {"document_read", "document_create", "document_replace"}.issubset(
        {tool.name for tool in document_tools()}
    )


@pytest.mark.asyncio
async def test_create_read_and_replace_document(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    await _document_create({"path": "notes/brief.md", "content": "Old draft\n"}, ctx)
    read = json.loads(await _document_read({"path": "notes/brief.md"}, ctx))
    assert read["content"] == "Old draft\n"
    assert read["sha256"] == hashlib.sha256(b"Old draft\n").hexdigest()
    await _document_replace(
        {"path": "notes/brief.md", "content": "Revised draft\n", "base_sha256": read["sha256"]},
        ctx,
    )
    assert (tmp_path / "notes/brief.md").read_text() == "Revised draft\n"


@pytest.mark.asyncio
async def test_document_create_never_overwrites(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("existing", encoding="utf-8")
    with pytest.raises(ToolError, match="already exists"):
        await _document_create({"path": "notes.txt", "content": "replacement"}, context(tmp_path))


@pytest.mark.asyncio
async def test_document_replace_rejects_stale_hash(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ToolError, match="SHA-256"):
        await _document_replace(
            {"path": "notes.md", "content": "new", "base_sha256": "0" * 64}, context(tmp_path)
        )


@pytest.mark.asyncio
async def test_document_path_cannot_escape_workspace(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="parent directories"):
        await _document_read({"path": "../outside.md"}, context(tmp_path))


@pytest.mark.asyncio
async def test_document_create_rejects_unsupported_extension(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="supported document formats"):
        await _document_create({"path": "payload.py", "content": "print(1)"}, context(tmp_path))

from __future__ import annotations

import hashlib
import json
from importlib import import_module
from pathlib import Path
from typing import Any

from eurus.errors import ToolError
from eurus.tools.base import Tool, ToolContext

MAX_DOCUMENT_BYTES = 2_000_000
TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".csv", ".json", ".html", ".xml"}
DOCUMENT_EXTENSIONS = TEXT_EXTENSIONS | {".docx", ".xlsx", ".pptx"}


def _resolve(context: ToolContext, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts:
        raise ToolError("document path must be relative to the workspace")
    root = context.workspace.resolve()
    cursor = root
    for part in candidate.parts:
        if part in {"", ".", ".."}:
            if part == "..":
                raise ToolError("document path cannot traverse parent directories")
            continue
        cursor = cursor / part
        if cursor.is_symlink():
            raise ToolError("symbolic links are not allowed in document paths")
    target = (root / candidate).resolve()
    if target != root and root not in target.parents:
        raise ToolError("document path is outside the configured workspace")
    if target.suffix.lower() not in DOCUMENT_EXTENSIONS:
        raise ToolError(
            "supported document formats: .md, .txt, .rst, .csv, .json, .html, .xml, "
            ".docx, .xlsx, .pptx"
        )
    return target


def _read_document(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ToolError("document does not exist or is not a regular file")
    if path.stat().st_size > MAX_DOCUMENT_BYTES:
        raise ToolError("document exceeds the 2 MB limit")
    if path.suffix.lower() == ".xlsx":
        try:
            openpyxl = import_module("openpyxl")
            workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
            rows = []
            for worksheet in workbook.worksheets:
                rows.append(f"## Sheet: {worksheet.title}")
                for row in worksheet.iter_rows(values_only=True):
                    rows.append(" | ".join("" if value is None else str(value) for value in row))
            workbook.close()
            return "\n".join(rows)
        except ImportError as exc:
            raise ToolError(
                "Excel support requires `pip install 'eurus-agent[documents]'`"
            ) from exc
        except Exception as exc:
            raise ToolError(f"cannot read Excel workbook: {type(exc).__name__}") from exc
    if path.suffix.lower() == ".pptx":
        try:
            pptx = import_module("pptx")
            presentation = pptx.Presentation(path)
            rows = []
            for index, slide in enumerate(presentation.slides, 1):
                rows.append(f"## Slide {index}")
                rows.extend(
                    shape.text
                    for shape in slide.shapes
                    if getattr(shape, "has_text_frame", False) and shape.text
                )
            return "\n".join(rows)
        except ImportError as exc:
            raise ToolError(
                "PowerPoint support requires `pip install 'eurus-agent[documents]'`"
            ) from exc
        except Exception as exc:
            raise ToolError(f"cannot read PowerPoint file: {type(exc).__name__}") from exc
    if path.suffix.lower() == ".docx":
        try:
            Document = import_module("docx").Document
        except ImportError as exc:
            raise ToolError(
                "DOCX support is optional; install `pip install 'eurus-agent[documents]'`"
            ) from exc
        try:
            document = Document(path)
            paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text]
            for table in document.tables:
                for row in table.rows:
                    paragraphs.append(" | ".join(cell.text for cell in row.cells))
            return "\n".join(paragraphs)
        except Exception as exc:
            raise ToolError(f"cannot read DOCX document: {type(exc).__name__}") from exc
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError("document is not valid UTF-8 text") from exc
    except OSError as exc:
        raise ToolError(f"cannot read document: {exc}") from exc


async def _document_read(arguments: dict[str, Any], context: ToolContext) -> str:
    path = _resolve(context, str(arguments["path"]))
    content = _read_document(path)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return json.dumps(
        {
            "path": path.relative_to(context.workspace.resolve()).as_posix(),
            "sha256": digest,
            "content": content,
        },
        ensure_ascii=False,
        indent=2,
    )


async def _document_create(arguments: dict[str, Any], context: ToolContext) -> str:
    path = _resolve(context, str(arguments["path"]))
    if path.suffix.lower() in {".xlsx", ".pptx"}:
        raise ToolError(
            "use excel_create or powerpoint_create for Office workbooks and presentations"
        )
    content = str(arguments["content"])
    if len(content.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise ToolError("document content exceeds the 2 MB limit")
    if path.exists():
        raise ToolError("document already exists; read it and use document_replace instead")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".docx":
            try:
                Document = import_module("docx").Document
            except ImportError as exc:
                raise ToolError(
                    "DOCX support is optional; install `pip install 'eurus-agent[documents]'`"
                ) from exc
            document = Document()
            for paragraph in content.splitlines() or [""]:
                document.add_paragraph(paragraph)
            document.save(path)
        else:
            path.write_text(content, encoding="utf-8")
    except ToolError:
        raise
    except OSError as exc:
        raise ToolError(f"cannot create document: {exc}") from exc
    relative = path.relative_to(context.workspace.resolve()).as_posix()
    return f"created {relative} ({len(content)} characters)"


async def _document_replace(arguments: dict[str, Any], context: ToolContext) -> str:
    path = _resolve(context, str(arguments["path"]))
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        raise ToolError("document_replace currently supports text-based documents only")
    current = _read_document(path)
    expected = str(arguments["base_sha256"]).strip().lower()
    actual = hashlib.sha256(current.encode("utf-8")).hexdigest()
    if len(expected) != 64 or expected != actual:
        raise ToolError("base_sha256 mismatch; read the latest document and retry with its SHA-256")
    replacement = str(arguments["content"])
    if len(replacement.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise ToolError("replacement exceeds the 2 MB limit")
    try:
        path.write_text(replacement, encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"cannot update document: {exc}") from exc
    relative = path.relative_to(context.workspace.resolve()).as_posix()
    return f"updated {relative} ({len(current)} -> {len(replacement)} characters)"


def document_tools() -> list[Tool]:
    path_schema = {"type": "string"}
    content_schema = {"type": "string"}
    return [
        Tool(
            "document_read",
            (
                "Read a workspace document (.md, .txt, .rst, .csv, .json, .html, .xml, or .docx). "
                "Returns its content and SHA-256 for safe revision workflows."
            ),
            {
                "type": "object",
                "properties": {"path": path_schema},
                "required": ["path"],
                "additionalProperties": False,
            },
            _document_read,
        ),
        Tool(
            "document_create",
            (
                "Create a new workspace document without overwriting an existing file. "
                "Requires approval."
            ),
            {
                "type": "object",
                "properties": {"path": path_schema, "content": content_schema},
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            _document_create,
            mutates=True,
            requires_approval=True,
        ),
        Tool(
            "document_replace",
            (
                "Replace a text document using the exact SHA-256 returned by document_read. "
                "Requires owner approval."
            ),
            {
                "type": "object",
                "properties": {
                    "path": path_schema,
                    "content": content_schema,
                    "base_sha256": {"type": "string"},
                },
                "required": ["path", "content", "base_sha256"],
                "additionalProperties": False,
            },
            _document_replace,
            mutates=True,
            requires_approval=True,
        ),
    ]

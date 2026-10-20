from __future__ import annotations

import json
from pathlib import Path

import pytest

from eurus.coding.document_tools import _document_read
from eurus.coding.office_tools import create_office_artifact, office_tools
from eurus.errors import ToolError
from eurus.tools.base import ToolContext


def test_office_tools_registered_with_approval() -> None:
    tools = {tool.name: tool for tool in office_tools()}
    assert {"word_create", "excel_create", "powerpoint_create"}.issubset(tools)
    assert all(tools[name].requires_approval for name in tools)
    assert all(tools[name].mutates for name in tools)


def test_word_create_generates_readable_docx(tmp_path: Path) -> None:
    result = create_office_artifact(
        ToolContext(workspace=tmp_path),
        "word",
        "reports/brief.docx",
        "# Executive summary\nA useful report.\n- First action\n- Second action",
    )
    assert "reports/brief.docx" in result
    content = (tmp_path / "reports/brief.docx").read_bytes()
    assert content.startswith(b"PK")


@pytest.mark.asyncio
async def test_excel_create_generates_workbook_and_escapes_formula_strings(tmp_path: Path) -> None:
    spec = {
        "title": "Budget",
        "sheets": [
            {
                "name": "Overview",
                "headers": ["Item", "Value"],
                "rows": [["Revenue", 100], ["=1+1", 2]],
            }
        ],
    }
    result = create_office_artifact(ToolContext(workspace=tmp_path), "excel", "budget.xlsx", spec)
    assert "budget.xlsx" in result
    import openpyxl

    workbook = openpyxl.load_workbook(tmp_path / "budget.xlsx", data_only=False)
    worksheet = workbook["Overview"]
    assert worksheet.freeze_panes == "A2"
    assert worksheet["A3"].value == "'=1+1"
    workbook.close()
    read = json.loads(
        await _document_read({"path": "budget.xlsx"}, ToolContext(workspace=tmp_path))
    )
    assert "Revenue" in read["content"]


def test_powerpoint_create_outputs_pptx_and_html_preview(tmp_path: Path) -> None:
    spec = {
        "title": "Quarterly Review",
        "subtitle": "Performance and next steps",
        "slides": [
            {
                "title": "What changed",
                "subtitle": "Quarterly view",
                "bullets": ["Growth", "Retention"],
                "callout": "Prioritize retention",
            }
        ],
    }
    result = create_office_artifact(
        ToolContext(workspace=tmp_path), "powerpoint", "quarterly.pptx", spec
    )
    assert "quarterly.pptx" in result
    assert "quarterly.preview.html" in result
    assert (tmp_path / "quarterly.pptx").read_bytes().startswith(b"PK")
    preview = (tmp_path / "quarterly.preview.html").read_text(encoding="utf-8")
    assert "<canvas" in preview
    assert "What changed" in preview


def test_office_create_does_not_overwrite(tmp_path: Path) -> None:
    (tmp_path / "existing.docx").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(ToolError, match="already exists"):
        create_office_artifact(ToolContext(workspace=tmp_path), "word", "existing.docx", "hello")


def test_excel_rejects_too_many_rows(tmp_path: Path) -> None:
    spec = {"sheets": [{"name": "Data", "headers": ["x"], "rows": [[1]] * 10001}]}
    with pytest.raises(ToolError, match="10000"):
        create_office_artifact(ToolContext(workspace=tmp_path), "excel", "large.xlsx", spec)


@pytest.mark.asyncio
async def test_office_replace_requires_current_document_hash(tmp_path: Path) -> None:
    from eurus.coding.office_tools import _office_replace

    context = ToolContext(workspace=tmp_path)
    create_office_artifact(context, "word", "draft.docx", "# Original\nFirst draft")
    read = json.loads(await _document_read({"path": "draft.docx"}, context))
    result = await _office_replace(
        {
            "path": "draft.docx",
            "format": "word",
            "base_sha256": read["sha256"],
            "content": "# Revised\nImproved draft",
        },
        context,
    )
    assert "updated draft.docx" in result
    revised = json.loads(await _document_read({"path": "draft.docx"}, context))
    assert "Improved draft" in revised["content"]


@pytest.mark.asyncio
async def test_office_replace_rejects_stale_hash(tmp_path: Path) -> None:
    from eurus.coding.office_tools import _office_replace

    context = ToolContext(workspace=tmp_path)
    create_office_artifact(context, "word", "draft.docx", "# Original\nFirst draft")
    with pytest.raises(ToolError, match="base_sha256 mismatch"):
        await _office_replace(
            {
                "path": "draft.docx",
                "format": "word",
                "base_sha256": "0" * 64,
                "content": "# Revised\nImproved draft",
            },
            context,
        )

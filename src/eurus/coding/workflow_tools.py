from __future__ import annotations

import json
from typing import Any

from eurus.coding.checks import check_tools
from eurus.coding.patchset import FilePatch, apply_patchset, preview_patchset
from eurus.coding.project_index import index_project
from eurus.errors import ToolError
from eurus.tools.base import Tool, ToolContext


def _parse_patches(value: Any) -> list[FilePatch]:
    if not isinstance(value, list) or not value:
        raise ToolError("patches must be a non-empty array")
    if len(value) > 20:
        raise ToolError("a patchset may contain at most 20 files")
    patches: list[FilePatch] = []
    for item in value:
        if not isinstance(item, dict):
            raise ToolError("each patch must be an object")
        path = item.get("path")
        expected = item.get("expected_content")
        replacement = item.get("new_content")
        if not isinstance(path, str) or not path:
            raise ToolError("each patch requires a non-empty path")
        if expected is not None and not isinstance(expected, str):
            raise ToolError("expected_content must be a string or null")
        if not isinstance(replacement, str):
            raise ToolError("new_content must be a string")
        patches.append(FilePatch(path, expected, replacement))
    return patches


async def _index(arguments: dict[str, Any], context: ToolContext) -> str:
    limit = int(arguments.get("max_files", 500))
    try:
        modules = index_project(context.workspace, max_files=limit)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    payload = [
        {
            "path": module.path,
            "imports": module.imports,
            "definitions": module.definitions,
            "parse_error": module.parse_error,
        }
        for module in modules
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


async def _preview(arguments: dict[str, Any], context: ToolContext) -> str:
    patches = _parse_patches(arguments.get("patches"))
    try:
        return preview_patchset(patches)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


async def _apply(arguments: dict[str, Any], context: ToolContext) -> str:
    patches = _parse_patches(arguments.get("patches"))
    try:
        result = apply_patchset(context.workspace, patches)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return json.dumps(
        {
            "status": "applied",
            "files": [
                {"path": change.path, "after_sha256": change.after_sha256}
                for change in result.changes
            ],
            "next_step": "Review the final diff and run the relevant tests.",
        },
        ensure_ascii=False,
        indent=2,
    )


def workflow_tools() -> list[Tool]:
    patch_item = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "expected_content": {"type": ["string", "null"]},
            "new_content": {"type": "string"},
        },
        "required": ["path", "expected_content", "new_content"],
        "additionalProperties": False,
    }
    patch_parameters = {
        "type": "object",
        "properties": {"patches": {"type": "array", "items": patch_item}},
        "required": ["patches"],
        "additionalProperties": False,
    }
    return [
        Tool(
            "index_project",
            "Index Python modules, imports, and definitions inside the configured workspace.",
            {
                "type": "object",
                "properties": {"max_files": {"type": "integer", "minimum": 1, "maximum": 20000}},
                "additionalProperties": False,
            },
            _index,
        ),
        Tool(
            "preview_patchset",
            "Preview a multi-file patchset without modifying files. Existing files require "
            "exact expected_content; use null for new files.",
            patch_parameters,
            _preview,
        ),
        Tool(
            "apply_patchset",
            "Apply a validated multi-file patchset. Requires one approval for the entire "
            "patchset; review the final diff afterward.",
            patch_parameters,
            _apply,
            mutates=True,
            requires_approval=True,
        ),
        *check_tools(),
    ]

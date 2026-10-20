from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path
from typing import Any

from eurus.coding.validation import run_validation
from eurus.errors import ToolError
from eurus.execution.host import scrubbed_env
from eurus.tools.base import Tool, ToolContext

CHECKS = ("tests", "types", "lint", "compile")


def _targets(workspace: Path, values: list[Any]) -> list[str]:
    root = workspace.resolve()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        path_part = text.split("::", 1)[0]
        if not text or text.startswith("-") or path_part.startswith("/"):
            raise ToolError(f"invalid target: {text!r}")
        resolved = (root / path_part).resolve()
        if resolved != root and root not in resolved.parents:
            raise ToolError(f"target is outside the workspace: {text}")
        if not resolved.exists():
            raise ToolError(f"target does not exist: {text}")
        result.append(text)
    return result


def _command(check: str, workspace: Path, targets: list[str]) -> tuple[str, ...]:
    if check == "tests":
        return (sys.executable, "-m", "pytest", "-q", "--tb=short", *targets)
    if check == "types":
        return (sys.executable, "-m", "mypy", *targets)
    if check == "lint":
        if shutil.which("ruff") is None:
            raise FileNotFoundError("ruff")
        return ("ruff", "check", *targets)
    defaults = [name for name in ("src", "tests") if (workspace / name).is_dir()] or ["."]
    return (sys.executable, "-m", "compileall", "-q", *(targets or defaults))


def _tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"... [{len(text) - limit} characters omitted] ...\n{text[-limit:]}"


async def _run_checks(arguments: dict[str, Any], context: ToolContext) -> str:
    names = list(dict.fromkeys(arguments.get("checks") or ["tests"]))
    targets = _targets(context.workspace, list(arguments.get("targets") or []))
    per_check = max(500, context.max_output // len(names))
    timeout = min(max(context.timeout, 1.0), 600.0)
    sections: list[str] = []
    failures = 0
    for name in names:
        try:
            argv = _command(name, context.workspace.resolve(), targets)
        except FileNotFoundError as exc:
            sections.append(f"[{name}] unavailable: {exc} is not installed")
            failures += 1
            continue
        result = await asyncio.to_thread(
            run_validation,
            context.workspace,
            argv,
            timeout_seconds=timeout,
            max_output_chars=min(max(per_check, 100), 200_000),
            env=scrubbed_env(),
        )
        if result.passed:
            status = "passed"
        elif result.timed_out:
            status = "timed out"
        else:
            status = f"failed (exit {result.returncode})"
        if not result.passed:
            failures += 1
        output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
        header = f"[{name}] {status} in {result.duration_seconds:.1f}s"
        sections.append(f"{header}\n{_tail(output, per_check)}".rstrip())
    sections.append("result: all checks passed" if not failures else f"result: {failures} failed")
    return "\n\n".join(sections)


def check_tools() -> list[Tool]:
    return [
        Tool(
            "run_checks",
            "Run project checks on the host with a reduced environment: tests (pytest), types "
            "(mypy), lint (ruff), compile. Optional targets are workspace paths, for example "
            "tests/test_x.py or tests/test_x.py::test_name. Read failures and fix the cause "
            "before running again.",
            {
                "type": "object",
                "properties": {
                    "checks": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(CHECKS)},
                    },
                    "targets": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
            _run_checks,
            requires_approval=True,
        )
    ]

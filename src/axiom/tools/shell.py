from __future__ import annotations

from typing import Any

from axiom.errors import ToolError
from axiom.execution.sandbox import ExecutionRequest, Sandbox
from axiom.tools.base import Tool, ToolContext


async def _run_shell(arguments: dict[str, Any], context: ToolContext, sandbox: Sandbox) -> str:
    command = str(arguments.get("command", "")).strip()
    if not command:
        raise ToolError("command must not be empty")
    if len(command) > 10000:
        raise ToolError("command exceeds 10000 characters")
    relative = str(arguments.get("cwd", ".")).strip() or "."
    if relative.startswith("/") or ".." in relative.split("/"):
        raise ToolError("cwd must be a relative path inside the workspace")
    cwd = (context.workspace / relative).resolve()
    if cwd != context.workspace.resolve() and context.workspace.resolve() not in cwd.parents:
        raise ToolError("cwd is outside the workspace")
    if not cwd.is_dir():
        raise ToolError("cwd does not exist")
    result = await sandbox.execute(
        ExecutionRequest(
            argv=("/bin/sh", "-lc", command),
            cwd="/workspace" if relative == "." else f"/workspace/{relative}",
            timeout_seconds=context.timeout,
            max_output_bytes=context.max_output,
        )
    )
    sections = []
    if result.stdout:
        sections.append(result.stdout.rstrip())
    if result.stderr:
        sections.append(f"stderr:\n{result.stderr.rstrip()}")
    sections.append(f"exit_code: {result.exit_code}")
    if result.timed_out:
        sections.append("timed_out: true")
    return "\n".join(sections)


def shell_tool(sandbox: Sandbox) -> Tool:
    async def handler(arguments: dict[str, Any], context: ToolContext) -> str:
        return await _run_shell(arguments, context, sandbox)

    return Tool(
        name="shell",
        description=(
            "Run a shell command inside the configured isolated container sandbox."
            if sandbox.isolated
            else "Run a shell command directly on the host. NOT isolated; the workspace is only "
            "the starting directory."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to run inside the sandbox."},
                "cwd": {
                    "type": "string",
                    "description": "Relative workspace directory; defaults to '.'",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        handler=handler,
        mutates=True,
        requires_approval=True,
    )

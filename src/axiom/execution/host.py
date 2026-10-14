from __future__ import annotations

import asyncio
import os
import shutil
import signal
from contextlib import suppress
from pathlib import Path

from axiom.errors import ToolError
from axiom.execution.sandbox import ExecutionRequest, ExecutionResult

SAFE_ENV = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TERM",
    "TMPDIR",
    "PREFIX",
    "VIRTUAL_ENV",
    "PYTHONPATH",
)


def scrubbed_env() -> dict[str, str]:
    return {name: os.environ[name] for name in SAFE_ENV if name in os.environ}


def _executable(argv0: str) -> str:
    if argv0 == "/bin/sh" and not Path(argv0).exists():
        return shutil.which("sh") or argv0
    return argv0


class HostSandbox:
    isolated: bool = False

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        if not request.argv or any(not isinstance(part, str) for part in request.argv):
            raise ToolError("command must be a non-empty argument list")
        if not self.workspace.is_dir():
            raise ToolError("workspace does not exist")
        relative = request.cwd.removeprefix("/workspace").lstrip("/")
        cwd = (self.workspace / relative).resolve()
        if cwd != self.workspace and self.workspace not in cwd.parents:
            raise ToolError("working directory is outside the workspace")
        if not cwd.is_dir():
            raise ToolError("working directory does not exist")

        argv = (_executable(request.argv[0]), *request.argv[1:])
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env=scrubbed_env(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=(os.name == "posix"),
            )
        except OSError as exc:
            raise ToolError(f"cannot start command: {exc}") from exc

        limit = request.max_output_bytes
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=min(request.timeout_seconds, 3600.0)
            )
        except (TimeoutError, asyncio.CancelledError) as exc:
            self._kill(process)
            stdout, stderr = await process.communicate()
            if isinstance(exc, asyncio.CancelledError):
                raise
            return ExecutionResult(
                stdout.decode("utf-8", errors="replace")[:limit],
                stderr.decode("utf-8", errors="replace")[:limit],
                124,
                timed_out=True,
            )
        return ExecutionResult(
            stdout.decode("utf-8", errors="replace")[:limit],
            stderr.decode("utf-8", errors="replace")[:limit],
            int(process.returncode or 0),
        )

    @staticmethod
    def _kill(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        with suppress(ProcessLookupError, PermissionError):
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()

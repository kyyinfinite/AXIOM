from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from axiom.errors import ToolError


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    argv: tuple[str, ...]
    cwd: str = "/workspace"
    timeout_seconds: float = 120.0
    max_output_bytes: int = 20000
    network_enabled: bool = False


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


class Sandbox(Protocol):
    isolated: bool

    async def execute(self, request: ExecutionRequest) -> ExecutionResult: ...


class ContainerSandbox:
    isolated: bool = True

    def __init__(
        self,
        workspace: Path,
        *,
        runtime: str = "docker",
        image: str = "python:3.12-alpine",
        memory_mb: int = 1024,
        cpu_limit: float = 1.0,
        pids_limit: int = 128,
        network_enabled: bool = False,
    ) -> None:
        self.workspace = workspace.resolve()
        self.runtime = runtime
        self.image = image
        self.memory_mb = memory_mb
        self.cpu_limit = cpu_limit
        self.pids_limit = pids_limit
        self.network_enabled = network_enabled

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        if not request.argv or any(not isinstance(part, str) for part in request.argv):
            raise ToolError("sandbox command must be a non-empty argument list")
        if not self.workspace.is_dir():
            raise ToolError("sandbox workspace does not exist")
        runtime_path = shutil.which(self.runtime)
        if runtime_path is None:
            raise ToolError(
                f"sandbox runtime '{self.runtime}' is unavailable; host fallback is disabled"
            )
        relative_cwd = request.cwd.removeprefix("/workspace").lstrip("/")
        cwd = (self.workspace / relative_cwd).resolve()
        if cwd != self.workspace and self.workspace not in cwd.parents:
            raise ToolError("working directory is outside the workspace")
        if not cwd.is_dir():
            raise ToolError("working directory does not exist")

        with tempfile.TemporaryDirectory(prefix="axiom-sandbox-") as temp_dir:
            cidfile = Path(temp_dir) / "container.id"
            network = "bridge" if (self.network_enabled and request.network_enabled) else "none"
            command = [
                runtime_path,
                "run",
                "--rm",
                "--cidfile",
                str(cidfile),
                "--network",
                network,
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                str(self.pids_limit),
                "--memory",
                f"{self.memory_mb}m",
                "--cpus",
                str(self.cpu_limit),
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=64m",
                "--mount",
                f"type=bind,src={self.workspace},dst=/workspace,rw",
                "--workdir",
                "/workspace",
                self.image,
                *request.argv,
            ]
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=(os.name == "posix"),
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=min(request.timeout_seconds, 3600.0)
                )
                return ExecutionResult(
                    stdout.decode("utf-8", errors="replace")[: request.max_output_bytes],
                    stderr.decode("utf-8", errors="replace")[: request.max_output_bytes],
                    int(process.returncode or 0),
                )
            except (TimeoutError, asyncio.CancelledError) as exc:
                with suppress(OSError, TimeoutError):
                    container_id = cidfile.read_text(encoding="utf-8").strip()
                    if container_id:
                        killer = await asyncio.create_subprocess_exec(
                            runtime_path,
                            "kill",
                            container_id,
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.DEVNULL,
                        )
                        await asyncio.wait_for(killer.wait(), timeout=5)
                if process.returncode is None:
                    process.kill()
                stdout, stderr = await process.communicate()
                if isinstance(exc, asyncio.CancelledError):
                    raise
                return ExecutionResult(
                    stdout.decode("utf-8", errors="replace")[: request.max_output_bytes],
                    stderr.decode("utf-8", errors="replace")[: request.max_output_bytes],
                    124,
                    timed_out=True,
                )

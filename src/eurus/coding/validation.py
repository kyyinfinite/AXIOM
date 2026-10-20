from __future__ import annotations

import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ALLOWED_MODULES = {"pytest", "compileall", "mypy"}
ALLOWED_BINARIES = {"ruff"}


@dataclass(frozen=True, slots=True)
class ValidationResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def validate_command(argv: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Validate a narrow, shell-free set of local Python quality commands."""
    if not argv or any(not isinstance(item, str) or not item for item in argv):
        raise ValueError("command must be a non-empty argument list")
    command = tuple(argv)
    executable = Path(command[0]).name
    is_versioned_python = (
        executable.startswith("python") and executable[6:].replace(".", "").isdigit()
    )
    if executable in {"python", "python3"} or is_versioned_python:
        if len(command) < 3 or command[1] != "-m" or command[2] not in ALLOWED_MODULES:
            raise ValueError("only python -m pytest|compileall|mypy commands are allowed")
        return command
    if executable == "ruff" and len(command) >= 2 and command[1] in {"check", "format"}:
        return command
    raise ValueError("command is not on the validation allowlist")


def run_validation(
    workspace: Path,
    argv: list[str] | tuple[str, ...],
    *,
    timeout_seconds: float = 60.0,
    max_output_chars: int = 20_000,
    env: Mapping[str, str] | None = None,
) -> ValidationResult:
    command = validate_command(argv)
    if not 1 <= timeout_seconds <= 600:
        raise ValueError("timeout_seconds must be between 1 and 600")
    if not 100 <= max_output_chars <= 200_000:
        raise ValueError("max_output_chars must be between 100 and 200000")
    root = workspace.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("workspace must be an existing directory")
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            shell=False,
            env=None if env is None else dict(env),
        )
        return ValidationResult(
            command,
            result.returncode,
            result.stdout[-max_output_chars:],
            result.stderr[-max_output_chars:],
            time.monotonic() - started,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return ValidationResult(
            command,
            124,
            str(stdout)[-max_output_chars:],
            str(stderr)[-max_output_chars:],
            time.monotonic() - started,
            timed_out=True,
        )

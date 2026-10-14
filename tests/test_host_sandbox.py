import asyncio
import os

import pytest

from axiom.config import load_config
from axiom.errors import ConfigError, ToolError
from axiom.execution.host import HostSandbox
from axiom.execution.sandbox import ExecutionRequest
from axiom.tools.shell import shell_tool


def run(sandbox, command, **kwargs):
    request = ExecutionRequest(argv=("/bin/sh", "-c", command), **kwargs)
    return asyncio.run(sandbox.execute(request))


def test_host_runs_command_in_workspace(tmp_path):
    result = run(HostSandbox(tmp_path), "pwd; echo err >&2; exit 3")
    assert result.stdout.strip() == str(tmp_path.resolve())
    assert result.stderr.strip() == "err"
    assert result.exit_code == 3


def test_host_does_not_pass_secrets(tmp_path):
    os.environ["AXIOM_TEST_API_KEY"] = "secret-value"
    try:
        result = run(HostSandbox(tmp_path), "echo ${AXIOM_TEST_API_KEY:-missing}")
    finally:
        del os.environ["AXIOM_TEST_API_KEY"]
    assert result.stdout.strip() == "missing"


def test_host_rejects_cwd_outside_workspace(tmp_path):
    with pytest.raises(ToolError, match="outside the workspace"):
        run(HostSandbox(tmp_path), "true", cwd="/workspace/../..")


def test_host_timeout_kills_process(tmp_path):
    result = run(HostSandbox(tmp_path), "sleep 30", timeout_seconds=0.5)
    assert result.timed_out
    assert result.exit_code == 124


def test_host_truncates_output(tmp_path):
    result = run(HostSandbox(tmp_path), "yes x | head -c 5000", max_output_bytes=100)
    assert len(result.stdout) == 100


def test_shell_tool_description_marks_host_as_not_isolated(tmp_path):
    assert "NOT isolated" in shell_tool(HostSandbox(tmp_path)).description


def write_config(tmp_path, body):
    path = tmp_path / "axiom.toml"
    path.write_text(body, encoding="utf-8")
    return load_config(config_path=path, environ={"AXIOM_HOME": str(tmp_path)})


def test_host_backend_requires_optout_of_required(tmp_path):
    with pytest.raises(ConfigError, match="required = false"):
        write_config(tmp_path, '[sandbox]\nbackend = "host"\n')
    config = write_config(tmp_path, '[sandbox]\nbackend = "host"\nrequired = false\n')
    assert config.sandbox.backend == "host"

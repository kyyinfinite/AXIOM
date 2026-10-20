import asyncio

import pytest

from eurus.config import load_config
from eurus.errors import ConfigError, ToolError
from eurus.execution.sandbox import ContainerSandbox, ExecutionRequest
from eurus.models import ToolCall
from eurus.security.approval import action_fingerprint, approval_request
from eurus.tools import Tool


async def noop(arguments, context):
    return "ok"


def test_sandbox_config_defaults(tmp_path):
    config = load_config(cwd=tmp_path, environ={"EURUS_HOME": str(tmp_path)})
    assert config.sandbox.backend == "container"
    assert config.sandbox.runtime == "docker"
    assert config.sandbox.network is False


def test_invalid_sandbox_config_rejected(tmp_path):
    path = tmp_path / "eurus.toml"
    path.write_text('[sandbox]\nbackend = "vm"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="supports only"):
        load_config(config_path=path, environ={"EURUS_HOME": str(tmp_path)})


def test_sandbox_never_falls_back_to_host(tmp_path):
    sandbox = ContainerSandbox(tmp_path, runtime="eurus-runtime-that-does-not-exist")
    with pytest.raises(ToolError, match="host fallback is disabled"):
        asyncio.run(sandbox.execute(ExecutionRequest(argv=("/bin/sh", "-lc", "echo unsafe"))))


def test_approval_fingerprint_is_stable_and_action_specific():
    first = ToolCall("call-1", "shell", {"command": "pwd"})
    same = ToolCall("call-2", "shell", {"command": "pwd"})
    changed = ToolCall("call-3", "shell", {"command": "rm -rf ."})
    assert action_fingerprint(first) == action_fingerprint(same)
    assert action_fingerprint(first) != action_fingerprint(changed)
    tool = Tool("shell", "shell", {"type": "object"}, noop, True, True)
    request = approval_request(first, tool)
    assert request.tool_name == "shell"
    assert request.action_fingerprint == action_fingerprint(first)

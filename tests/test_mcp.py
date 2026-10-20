from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eurus.mcp.manager import MCPServerConfig, MCPToolManager
from eurus.models import ToolCall
from eurus.tools.base import ToolContext, ToolRegistry


def test_stdio_config_round_trip() -> None:
    config = MCPServerConfig(
        name="filesystem",
        command="python",
        args=("server.py", "--readonly"),
        env={"MODE": "safe"},
    )
    assert MCPServerConfig.from_dict(config.to_dict()) == config


def test_http_config_requires_tls() -> None:
    with pytest.raises(ValueError, match="https://"):
        MCPServerConfig(name="remote", transport="http", url="http://localhost/mcp")


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown MCP server fields"):
        MCPServerConfig.from_dict({"name": "x", "command": "python", "extra": True})


def test_save_load_config_is_private_and_atomic(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    config = MCPServerConfig(name="filesystem", command="python", args=("server.py",))
    MCPToolManager.save(path, [config])
    manager = MCPToolManager.load(path)
    assert manager.configs == {"filesystem": config}
    assert json.loads(path.read_text(encoding="utf-8"))["servers"][0]["name"] == "filesystem"
    assert not path.with_suffix(".json.tmp").exists()


def test_exposed_names_are_prefixed_and_bounded() -> None:
    name = MCPToolManager._exposed_name("remote server", "read file/contents")
    assert name.startswith("mcp_remote_server_read_file_contents")
    assert len(MCPToolManager._exposed_name("x" * 48, "y" * 80)) <= 64


def test_remote_tools_require_approval_and_execute() -> None:
    class FakeSession:
        async def call_tool(self, name: str, arguments: dict[str, object]) -> object:
            assert name == "remote_read"
            assert arguments == {"path": "README.md"}
            return SimpleNamespace(content=[SimpleNamespace(text="read ok")], isError=False)

    config = MCPServerConfig(name="remote", command="fake")
    manager = MCPToolManager([config])
    manager._connected = True
    manager._sessions["remote"] = FakeSession()
    manager._tools["mcp_remote_read"] = {
        "server": "remote",
        "remote_name": "remote_read",
        "description": "Read a file",
        "schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }
    registry = ToolRegistry()
    assert manager.register_tools(registry) == 1
    tool = registry.get("mcp_remote_read")
    assert tool is not None
    assert tool.mutates is True
    assert tool.requires_approval is True
    import asyncio

    result = asyncio.run(
        registry.execute(
            ToolCall("call-1", "mcp_remote_read", {"path": "README.md"}),
            ToolContext(Path.cwd()),
        )
    )
    assert result.output == "read ok"
    assert result.is_error is False


def test_missing_config_loads_empty_manager(tmp_path: Path) -> None:
    assert MCPToolManager.load(tmp_path / "missing.json").configs == {}


@pytest.mark.asyncio
async def test_empty_manager_connect_does_not_require_optional_sdk() -> None:
    manager = MCPToolManager()
    await manager.connect()
    assert manager.connected is True
    assert manager.status() == []
    await manager.close()

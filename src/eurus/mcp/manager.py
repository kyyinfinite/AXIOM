from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eurus.errors import ToolError
from eurus.tools.base import Tool, ToolContext, ToolRegistry

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_SCHEMA_BYTES = 64_000


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    """Connection settings for one explicitly configured MCP server."""

    name: str
    transport: str = "stdio"
    command: str = ""
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", self.name):
            raise ValueError("server name must use 1-48 letters, digits, '_' or '-'")
        if self.transport not in {"stdio", "http"}:
            raise ValueError("transport must be 'stdio' or 'http'")
        if self.transport == "stdio" and not self.command.strip():
            raise ValueError("stdio transport requires a command")
        if self.transport == "http" and not self.url.startswith("https://"):
            raise ValueError("HTTP MCP servers must use an https:// URL")
        if not 1 <= self.timeout_seconds <= 300:
            raise ValueError("timeout_seconds must be between 1 and 300")
        if len(self.env) > 100 or len(self.headers) > 100:
            raise ValueError("too many environment variables or HTTP headers")
        if any("\x00" in key or "\x00" in value for key, value in self.env.items()):
            raise ValueError("environment variables cannot contain NUL characters")
        if any(
            "\r" in key or "\n" in key or "\r" in value or "\n" in value
            for key, value in self.headers.items()
        ):
            raise ValueError("HTTP headers cannot contain newline characters")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCPServerConfig:
        allowed = {
            "name",
            "transport",
            "command",
            "args",
            "env",
            "url",
            "headers",
            "enabled",
            "timeout_seconds",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown MCP server fields: {', '.join(sorted(unknown))}")
        return cls(
            name=str(data.get("name", "")),
            transport=str(data.get("transport", "stdio")),
            command=str(data.get("command", "")),
            args=tuple(str(item) for item in data.get("args", [])),
            env={str(key): str(value) for key, value in data.get("env", {}).items()},
            url=str(data.get("url", "")),
            headers={str(key): str(value) for key, value in data.get("headers", {}).items()},
            enabled=bool(data.get("enabled", True)),
            timeout_seconds=float(data.get("timeout_seconds", 30.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "transport": self.transport,
            "command": self.command,
            "args": list(self.args),
            "env": dict(self.env),
            "url": self.url,
            "headers": dict(self.headers),
            "enabled": self.enabled,
            "timeout_seconds": self.timeout_seconds,
        }


class MCPToolManager:
    """Own MCP connections and expose discovered tools through Eurus' registry.

    All remote tools require owner approval by default. The optional MCP SDK is
    imported lazily so the core package remains usable without MCP installed.
    """

    def __init__(self, configs: list[MCPServerConfig] | None = None) -> None:
        self.configs = {item.name: item for item in configs or []}
        if len(self.configs) != len(configs or []):
            raise ValueError("duplicate MCP server name")
        self._stack = AsyncExitStack()
        self._sessions: dict[str, Any] = {}
        self._tools: dict[str, dict[str, Any]] = {}
        self._connected = False

    @staticmethod
    def load(path: Path) -> MCPToolManager:
        if not path.exists():
            return MCPToolManager()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read MCP config {path}: {exc}") from exc
        if not isinstance(raw, dict) or set(raw) != {"servers"}:
            raise ValueError("MCP config must contain only a 'servers' array")
        if not isinstance(raw["servers"], list):
            raise ValueError("MCP config 'servers' must be an array")
        return MCPToolManager([MCPServerConfig.from_dict(item) for item in raw["servers"]])

    @staticmethod
    def save(path: Path, configs: list[MCPServerConfig]) -> None:
        names = [item.name for item in configs]
        if len(names) != len(set(names)):
            raise ValueError("duplicate MCP server name")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = path.with_suffix(path.suffix + ".tmp")
        payload = {"servers": [item.to_dict() for item in configs]}
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        with suppress(OSError):
            os.chmod(temporary, 0o600)
        temporary.replace(path)

    async def connect(self) -> None:
        if self._connected:
            return
        if not any(config.enabled for config in self.configs.values()):
            self._connected = True
            return
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as exc:
            raise RuntimeError(
                "MCP support is optional; install Eurus with the 'mcp' extra"
            ) from exc

        try:
            for config in self.configs.values():
                if not config.enabled:
                    continue
                if config.transport == "stdio":
                    parameters = StdioServerParameters(
                        command=config.command,
                        args=list(config.args),
                        env=config.env or None,
                    )
                    read_stream, write_stream = await self._stack.enter_async_context(
                        stdio_client(parameters)
                    )
                else:
                    read_stream, write_stream, _session_id = await self._stack.enter_async_context(
                        streamablehttp_client(config.url, headers=config.headers or None)
                    )
                session = await self._stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await asyncio.wait_for(session.initialize(), timeout=config.timeout_seconds)
                response = await asyncio.wait_for(
                    session.list_tools(), timeout=config.timeout_seconds
                )
                discovered: dict[str, Any] = {}
                for item in response.tools:
                    if not item.name or not isinstance(item.name, str):
                        continue
                    schema = item.inputSchema or {"type": "object", "properties": {}}
                    if len(json.dumps(schema).encode("utf-8")) > _MAX_SCHEMA_BYTES:
                        continue
                    exposed = self._exposed_name(config.name, item.name)
                    if exposed in self._tools:
                        raise ValueError(f"duplicate exposed MCP tool name: {exposed}")
                    discovered[exposed] = {
                        "server": config.name,
                        "remote_name": item.name,
                        "description": str(item.description or "")[:2000],
                        "schema": schema,
                    }
                self._sessions[config.name] = session
                self._tools.update(discovered)
            self._connected = True
        except Exception:
            await self.close()
            raise

    @staticmethod
    def _exposed_name(server_name: str, tool_name: str) -> str:
        safe_server = _SAFE_NAME.sub("_", server_name).strip("_") or "server"
        safe_tool = _SAFE_NAME.sub("_", tool_name).strip("_") or "tool"
        name = f"mcp_{safe_server}_{safe_tool}"
        return name[:64]

    def register_tools(self, registry: ToolRegistry) -> int:
        if not self._connected:
            raise RuntimeError("connect to MCP servers before registering their tools")
        count = 0
        for name, item in self._tools.items():
            if name in registry:
                raise ValueError(f"MCP tool conflicts with existing tool: {name}")
            registry.register(
                Tool(
                    name=name,
                    description=f"MCP server {item['server']}: {item['description']}",
                    parameters=item["schema"],
                    handler=self._handler(name),
                    mutates=True,
                    requires_approval=True,
                )
            )
            count += 1
        return count

    def _handler(
        self, exposed_name: str
    ) -> Callable[[dict[str, Any], ToolContext], Awaitable[str]]:
        async def call(arguments: dict[str, Any], context: ToolContext) -> str:
            item = self._tools[exposed_name]
            session = self._sessions.get(item["server"])
            if session is None:
                raise ToolError(f"MCP server '{item['server']}' is not connected")
            config = self.configs[item["server"]]
            try:
                result = await asyncio.wait_for(
                    session.call_tool(item["remote_name"], arguments=arguments),
                    timeout=min(context.timeout, config.timeout_seconds),
                )
            except TimeoutError as exc:
                raise ToolError(f"MCP tool timed out after {config.timeout_seconds:g}s") from exc
            except Exception as exc:
                raise ToolError(f"MCP call failed: {type(exc).__name__}: {exc}") from exc
            content: list[str] = []
            for block in getattr(result, "content", []):
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    content.append(text)
                else:
                    content.append(f"[{getattr(block, 'type', 'content')} content omitted]")
            output = "\n".join(content) or json.dumps(getattr(result, "structuredContent", {}))
            if getattr(result, "isError", False):
                raise ToolError(output or "MCP server returned an error")
            return output

        return call

    @property
    def connected(self) -> bool:
        return self._connected

    def status(self) -> list[dict[str, Any]]:
        rows = []
        for name, config in self.configs.items():
            rows.append(
                {
                    "name": name,
                    "transport": config.transport,
                    "enabled": config.enabled,
                    "connected": name in self._sessions,
                    "tools": sum(item["server"] == name for item in self._tools.values()),
                }
            )
        return rows

    async def close(self) -> None:
        await self._stack.aclose()
        self._stack = AsyncExitStack()
        self._sessions.clear()
        self._tools.clear()
        self._connected = False

    async def __aenter__(self) -> MCPToolManager:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.close()

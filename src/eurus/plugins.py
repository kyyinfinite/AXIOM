from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from eurus.tools.base import Tool, ToolRegistry

_PLUGIN_ID = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
_VERSION = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+][0-9A-Za-z.-]+)?$")
_CAPABILITY = re.compile(r"^[a-z][a-z0-9_.:-]{0,63}$")


@dataclass(frozen=True, slots=True)
class PluginManifest:
    plugin_id: str
    version: str
    description: str
    capabilities: tuple[str, ...] = ()
    required_config: tuple[str, ...] = ()
    required_permissions: tuple[str, ...] = ()
    min_eurus_version: str = "0.0.0"
    max_eurus_version: str | None = None

    def validate(self) -> None:
        if not _PLUGIN_ID.fullmatch(self.plugin_id):
            raise ValueError("invalid plugin identifier")
        if not _VERSION.fullmatch(self.version):
            raise ValueError("plugin version must be semantic-version-like")
        if not self.description.strip() or len(self.description) > 500:
            raise ValueError("invalid plugin description")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("duplicate plugin capability")
        if any(not _CAPABILITY.fullmatch(value) for value in self.capabilities):
            raise ValueError("invalid plugin capability")
        if len(set(self.required_config)) != len(self.required_config):
            raise ValueError("duplicate required configuration key")
        if len(set(self.required_permissions)) != len(self.required_permissions):
            raise ValueError("duplicate required permission")
        if not _VERSION.fullmatch(self.min_eurus_version):
            raise ValueError("invalid minimum Eurus version")
        if self.max_eurus_version and not _VERSION.fullmatch(self.max_eurus_version):
            raise ValueError("invalid maximum Eurus version")


class Plugin(Protocol):
    manifest: PluginManifest

    def tools(self) -> Iterable[Tool]: ...

    def start(self, config: dict[str, str]) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class PluginStatus:
    plugin_id: str
    version: str
    enabled: bool
    healthy: bool
    error: str = ""
    tools: tuple[str, ...] = field(default_factory=tuple)


class PluginManager:
    def __init__(self, database: Path, *, eurus_version: str = "0.0.0") -> None:
        self.database = database.expanduser()
        self.database.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.eurus_version = eurus_version
        self._plugins: dict[str, Plugin] = {}
        self._registries: dict[str, ToolRegistry] = {}
        self._status: dict[str, PluginStatus] = {}
        with self._connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS plugin_state ("
                "plugin_id TEXT PRIMARY KEY, enabled INTEGER NOT NULL, updated_at TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=10)
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _enabled(self, plugin_id: str) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT enabled FROM plugin_state WHERE plugin_id=?", (plugin_id,)
            ).fetchone()
        return bool(row and row[0])

    def enable(
        self, plugin: Plugin, registry: ToolRegistry, config: dict[str, str] | None = None
    ) -> PluginStatus:
        manifest = plugin.manifest
        manifest.validate()
        if manifest.plugin_id in self._plugins:
            raise ValueError(f"plugin already loaded: {manifest.plugin_id}")
        if not _version_in_range(
            self.eurus_version, manifest.min_eurus_version, manifest.max_eurus_version
        ):
            raise ValueError("plugin is incompatible with this Eurus version")
        settings = config or {}
        missing = sorted(set(manifest.required_config) - settings.keys())
        if missing:
            raise ValueError("missing required plugin configuration: " + ", ".join(missing))
        tools = tuple(plugin.tools())
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("plugin declares duplicate tool names")
        collisions = sorted(set(names).intersection(registry.names()))
        if collisions:
            raise ValueError("plugin tool name collision: " + ", ".join(collisions))
        try:
            plugin.start(dict(settings))
            for tool in tools:
                registry.register(tool)
        except Exception as exc:
            with suppress(Exception):
                plugin.close()
            status = PluginStatus(
                manifest.plugin_id,
                manifest.version,
                False,
                False,
                _safe_error(exc),
                tuple(names),
            )
            self._status[manifest.plugin_id] = status
            raise RuntimeError(f"plugin failed to start: {status.error}") from exc
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO plugin_state(plugin_id, enabled, updated_at) VALUES (?, 1, ?) "
                "ON CONFLICT(plugin_id) DO UPDATE SET enabled=1, updated_at=excluded.updated_at",
                (manifest.plugin_id, datetime.now(UTC).isoformat()),
            )
        self._plugins[manifest.plugin_id] = plugin
        self._registries[manifest.plugin_id] = registry
        status = PluginStatus(manifest.plugin_id, manifest.version, True, True, tools=tuple(names))
        self._status[manifest.plugin_id] = status
        return status

    def disable(self, plugin_id: str) -> PluginStatus:
        plugin = self._plugins.pop(plugin_id, None)
        registry = self._registries.pop(plugin_id, None)
        current = self._status.get(plugin_id)
        if plugin is None and current is None:
            raise KeyError(plugin_id)
        error = ""
        if plugin is not None:
            if registry is not None and current is not None:
                for tool_name in current.tools:
                    registry.unregister(tool_name)
            try:
                plugin.close()
            except Exception as exc:
                error = _safe_error(exc)
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO plugin_state(plugin_id, enabled, updated_at) VALUES (?, 0, ?) "
                "ON CONFLICT(plugin_id) DO UPDATE SET enabled=0, updated_at=excluded.updated_at",
                (plugin_id, datetime.now(UTC).isoformat()),
            )
        status = PluginStatus(
            plugin_id,
            current.version if current else "unknown",
            False,
            not bool(error),
            error,
            current.tools if current else (),
        )
        self._status[plugin_id] = status
        return status

    def list(self) -> tuple[PluginStatus, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT plugin_id, enabled FROM plugin_state ORDER BY plugin_id"
            ).fetchall()
        result: list[PluginStatus] = []
        for plugin_id, enabled in rows:
            status = self._status.get(plugin_id)
            if status:
                result.append(status)
            else:
                result.append(
                    PluginStatus(plugin_id, "unknown", bool(enabled), False, "not loaded")
                )
        return tuple(result)

    def inspect(self, plugin_id: str) -> PluginStatus:
        status = self._status.get(plugin_id)
        if status is None:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT enabled FROM plugin_state WHERE plugin_id=?", (plugin_id,)
                ).fetchone()
            if row is None:
                raise KeyError(plugin_id)
            return PluginStatus(plugin_id, "unknown", bool(row[0]), False, "plugin is not loaded")
        return status


def _version_tuple(value: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        raise ValueError("invalid version")
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _version_in_range(current: str, minimum: str, maximum: str | None) -> bool:
    version = _version_tuple(current)
    return version >= _version_tuple(minimum) and (
        maximum is None or version <= _version_tuple(maximum)
    )


def _safe_error(error: Exception) -> str:
    text = re.sub(
        r"(?i)(token|api[_ -]?key|password|secret)\s*[:=]\s*\S+",
        r"\1=[REDACTED]",
        str(error),
    )
    return f"{type(error).__name__}: {text[:300]}"

from pathlib import Path

import pytest

from eurus.plugins import PluginManager, PluginManifest
from eurus.tools.base import Tool, ToolContext, ToolRegistry


async def _handler(arguments: dict, context: ToolContext) -> str:
    return "ok"


class DemoPlugin:
    manifest = PluginManifest("demo-plugin", "1.0.0", "Test plugin", ("files.read",))

    def __init__(self) -> None:
        self.started = False
        self.closed = False

    def tools(self):
        return (Tool("demo_read", "Read demo", {"type": "object"}, _handler),)

    def start(self, config: dict[str, str]) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True


def test_plugin_enable_disable_and_persisted_state(tmp_path: Path) -> None:
    manager = PluginManager(tmp_path / "plugins.db", eurus_version="1.2.0")
    registry = ToolRegistry()
    plugin = DemoPlugin()
    status = manager.enable(plugin, registry)
    assert status.enabled and status.healthy
    assert "demo_read" in registry
    disabled = manager.disable("demo-plugin")
    assert not disabled.enabled and plugin.closed
    assert "demo_read" not in registry
    assert manager.list()[0].enabled is False


def test_plugin_rejects_tool_collision(tmp_path: Path) -> None:
    manager = PluginManager(tmp_path / "plugins.db")
    registry = ToolRegistry([Tool("demo_read", "Existing", {"type": "object"}, _handler)])
    with pytest.raises(ValueError, match="collision"):
        manager.enable(DemoPlugin(), registry)


def test_plugin_requires_explicit_config_and_compatibility(tmp_path: Path) -> None:
    manager = PluginManager(tmp_path / "plugins.db", eurus_version="1.0.0")

    class ConfigPlugin(DemoPlugin):
        manifest = PluginManifest(
            "config-plugin", "1.0.0", "Config plugin", required_config=("endpoint",)
        )

    with pytest.raises(ValueError, match="missing required"):
        manager.enable(ConfigPlugin(), ToolRegistry())

    class NewPlugin(DemoPlugin):
        manifest = PluginManifest("new-plugin", "1.0.0", "New plugin", min_eurus_version="2.0.0")

    with pytest.raises(ValueError, match="incompatible"):
        manager.enable(NewPlugin(), ToolRegistry())


def test_plugin_start_failure_closes_plugin(tmp_path: Path) -> None:
    class FailingPlugin(DemoPlugin):
        def start(self, config: dict[str, str]) -> None:
            raise RuntimeError("startup failed")

    manager = PluginManager(tmp_path / "plugins.db")
    plugin = FailingPlugin()

    with pytest.raises(RuntimeError, match="plugin failed to start"):
        manager.enable(plugin, ToolRegistry())

    assert plugin.closed
    status = manager.inspect("demo-plugin")
    assert not status.enabled
    assert not status.healthy

from pathlib import Path

import pytest

from eurus.coding.office_plugin import OfficePlugin, office_plugin
from eurus.coding.tools import coding_tools
from eurus.plugins import PluginManager
from eurus.tools.base import ToolRegistry


def test_office_plugin_exposes_existing_approved_tools() -> None:
    plugin = office_plugin()
    tools = {tool.name: tool for tool in plugin.tools()}
    assert set(tools) == {
        "word_create",
        "excel_create",
        "powerpoint_create",
        "office_replace",
    }
    assert all(tool.mutates and tool.requires_approval for tool in tools.values())
    assert plugin.manifest.plugin_id == "office-artifacts"
    assert {"office.word", "office.excel", "office.powerpoint"}.issubset(
        plugin.manifest.capabilities
    )


def test_office_plugin_can_be_enabled_and_disabled(tmp_path: Path) -> None:
    manager = PluginManager(tmp_path / "plugins.db", eurus_version="1.0.0")
    registry = ToolRegistry()
    plugin = OfficePlugin()

    enabled = manager.enable(plugin, registry)
    assert enabled.enabled and enabled.healthy
    assert set(enabled.tools) == {
        "word_create",
        "excel_create",
        "powerpoint_create",
        "office_replace",
    }
    assert set(enabled.tools).issubset(registry.names())

    disabled = manager.disable("office-artifacts")
    assert not disabled.enabled and disabled.healthy
    assert not set(enabled.tools).intersection(registry.names())
    assert manager.inspect("office-artifacts").enabled is False


def test_office_plugin_is_not_registered_twice_in_default_coding_registry(
    tmp_path: Path,
) -> None:
    names = {tool.name for tool in coding_tools()}
    plugin_names = {tool.name for tool in office_plugin().tools()}
    assert plugin_names.issubset(names)
    registry = ToolRegistry(coding_tools())
    manager = PluginManager(tmp_path / "plugins.db")
    with pytest.raises(ValueError, match="collision"):
        manager.enable(office_plugin(), registry)

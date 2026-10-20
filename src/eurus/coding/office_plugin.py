from __future__ import annotations

from eurus.coding.office_tools import office_tools
from eurus.plugins import PluginManifest
from eurus.tools.base import Tool


class OfficePlugin:
    manifest = PluginManifest(
        plugin_id="office-artifacts",
        version="1.0.0",
        description="Creates and revises Word, Excel, and PowerPoint artifacts.",
        capabilities=("office.word", "office.excel", "office.powerpoint", "office.review"),
    )

    def __init__(self) -> None:
        self._active = False

    def tools(self) -> tuple[Tool, ...]:
        return tuple(office_tools())

    def start(self, config: dict[str, str]) -> None:
        self._active = True

    def close(self) -> None:
        self._active = False


def office_plugin() -> OfficePlugin:
    return OfficePlugin()

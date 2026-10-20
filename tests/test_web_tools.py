from __future__ import annotations

import json

import pytest

from eurus.agent import web_tools as web
from eurus.errors import ToolError
from eurus.tools.builtin import default_registry


def test_web_tools_are_registered_with_capability_tools() -> None:
    registry = default_registry(include_capability_tools=True)
    assert {"web_search", "web_fetch", "deep_search"}.issubset(set(registry.names()))


def test_validate_url_rejects_localhost_and_private_ip() -> None:
    with pytest.raises(ToolError, match="local and private"):
        web._validate_url("http://127.0.0.1/admin")
    with pytest.raises(ToolError, match="local and private"):
        web._validate_url("http://localhost:8000")


def test_validate_url_rejects_non_http_schemes() -> None:
    with pytest.raises(ToolError, match="http or https"):
        web._validate_url("file:///etc/passwd")


def test_extract_fallback_removes_script_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "trafilatura", None)
    page = web._extract(
        "<html><body><h1>Title</h1><script>secret()</script><p>Useful text</p></body></html>",
        "https://example.com",
    )
    assert "Useful text" in page["text"]
    assert "secret" not in page["text"]


@pytest.mark.asyncio
async def test_search_web_formats_search_results(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_to_thread(function, *args):
        assert function is web._search_sync
        assert args == ("python web research", 3)
        return [{"title": "Docs", "url": "https://example.com/docs", "snippet": "Useful docs"}]

    monkeypatch.setattr(web.asyncio, "to_thread", fake_to_thread)
    output = await web._search_web({"query": "python web research", "limit": 3}, None)  # type: ignore[arg-type]
    data = json.loads(output)
    assert data["results"][0]["url"] == "https://example.com/docs"


@pytest.mark.asyncio
async def test_search_web_rejects_invalid_limit() -> None:
    with pytest.raises(ToolError, match="limit must be between"):
        await web._search_web({"query": "test", "limit": 20}, None)  # type: ignore[arg-type]

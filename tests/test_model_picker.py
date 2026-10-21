import asyncio

import pytest
from rich.console import Console

from eurus.catalog import BUILTIN_PROVIDERS, ProviderSpec
from eurus.config import load_config
from eurus.providers.discovery import (
    Discovery,
    ModelInfo,
    discover,
    listing_request,
    parse_models,
)
from eurus.tui.picker import (
    filter_models,
    page_bounds,
    parse_choice,
    provider_rows,
    with_active_first,
)
from eurus.tui.renderer import TerminalRenderer
from eurus.tui.session import InteractiveSession


def spec(protocol="openai", env="K", base="https://api.example.com/v1"):
    return ProviderSpec("p", protocol, base, env)


def test_openai_listing_filters_non_chat_models():
    payload = {
        "data": [
            {"id": "llama-3.3-70b", "context_length": 131072},
            {"id": "whisper-large-v3"},
            {"id": "text-embedding-3-small"},
            {"id": "llama-3.3-70b"},
        ]
    }
    models = parse_models("openai", payload)
    assert [m.id for m in models] == ["llama-3.3-70b"] and models[0].context == 131072


def test_gemini_listing_keeps_generate_content_only():
    payload = {
        "models": [
            {"name": "models/gemini-2.0-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]},
        ]
    }
    assert [m.id for m in parse_models("gemini", payload)] == ["gemini-2.0-flash"]


def test_anthropic_and_gemini_use_header_auth_not_url():
    url, headers = listing_request(spec("anthropic", base="https://api.anthropic.com"), "secret")
    assert url.endswith("/v1/models?limit=1000") and headers["x-api-key"] == "secret"
    url, headers = listing_request(spec("gemini", base="https://g.example.com"), "secret")
    assert "secret" not in url and headers["x-goog-api-key"] == "secret"
    url, headers = listing_request(spec(), "secret")
    assert url == "https://api.example.com/v1/models"
    assert headers["Authorization"] == "Bearer secret"


def test_discover_reports_missing_key_without_network():
    async def boom(url, headers):
        raise AssertionError("no request expected")

    result = asyncio.run(discover(spec(), None, boom))
    assert result.error == "no API key" and not result.models


def test_discover_handles_status_and_success():
    async def rejected(url, headers):
        return 401, {}

    async def good(url, headers):
        return 200, {"data": [{"id": "b"}, {"id": "a"}]}

    assert "key rejected" in asyncio.run(discover(spec(), "k", rejected)).error
    assert [m.id for m in asyncio.run(discover(spec(), "k", good)).models] == ["a", "b"]


def test_discover_retries_user_agent_on_403():
    seen = []

    async def edge(url, headers):
        seen.append(headers["User-Agent"])
        if len(seen) == 1:
            return 403, None
        return 200, {"data": [{"id": "m"}]}

    result = asyncio.run(discover(spec(), "k", edge))
    assert len(seen) == 2 and seen[0] != seen[1] and [m.id for m in result.models] == ["m"]


def test_choice_parsing():
    assert parse_choice("", 5).kind == "quit"
    assert parse_choice("N", 5).kind == "next" and parse_choice("p", 5).kind == "prev"
    assert parse_choice("3", 5).index == 2
    assert parse_choice("9", 5).kind == "text"
    assert parse_choice("llama", 5).text == "llama"


def test_paging_and_filtering():
    assert page_bounds(0, 3, 8) == (0, 1, 0, 0)
    assert page_bounds(20, 9, 8) == (2, 3, 16, 20)
    models = [ModelInfo("alpha"), ModelInfo("beta", "Llama Beta")]
    assert [m.id for m in filter_models(models, "LLAMA")] == ["beta"]
    assert [m.id for m in with_active_first(models, "beta")] == ["beta", "alpha"]


def only_keys(monkeypatch, **keys):
    for provider in BUILTIN_PROVIDERS.values():
        if provider.api_key_env:
            monkeypatch.delenv(provider.api_key_env, raising=False)
    for name, value in keys.items():
        monkeypatch.setenv(name, value)


def build_session(tmp_path, answers, model="groq/llama", env=None):
    environ = {"EURUS_HOME": str(tmp_path), **(env or {})}
    config = load_config(cwd=tmp_path, environ=environ, overrides={"agent": {"model": model}})
    console = Console(record=True, width=40, force_terminal=False)
    renderer = TerminalRenderer(console)
    queue = list(answers)
    renderer.ask = lambda prompt: queue.pop(0)
    return InteractiveSession(config, renderer), console


def test_provider_rows_mark_ready_and_active(tmp_path, monkeypatch):
    only_keys(monkeypatch, GROQ_API_KEY="gsk_abcdefghij")
    config = load_config(
        cwd=tmp_path,
        environ={"EURUS_HOME": str(tmp_path)},
        overrides={"agent": {"model": "groq/llama"}},
    )
    rows = {row.name: row for row in provider_rows(config)}
    assert rows["groq"].ready and rows["groq"].active
    assert not rows["openai"].ready and rows["openai"].note == "OPENAI_API_KEY"


def test_picker_switches_model_and_marks_active(tmp_path, monkeypatch):
    only_keys(monkeypatch, GROQ_API_KEY="gsk_abcdefghij")

    async def fake(spec, key):
        assert key == "gsk_abcdefghij"
        return Discovery(spec.name, (ModelInfo("llama"), ModelInfo("mixtral", "", 32768)))

    monkeypatch.setattr("eurus.tui.session.discover", fake)
    session, console = build_session(tmp_path, ["1", "2"])
    asyncio.run(session.submit("/model"))
    output = console.export_text()
    assert "PROVIDERS" in output and "GROQ MODELS" in output and "gsk_" not in output
    assert session.config.agent.model == "groq/mixtral"
    assert "Active model: groq/mixtral" in output


def test_picker_manual_entry_filter_and_cancel(tmp_path, monkeypatch):
    only_keys(monkeypatch, GROQ_API_KEY="gsk_abcdefghij")

    async def failing(spec, key):
        return Discovery(spec.name, error="network error: ConnectError")

    monkeypatch.setattr("eurus.tui.session.discover", failing)
    session, console = build_session(tmp_path, ["groq", "=my-model"])
    asyncio.run(session.submit("/model"))
    assert "network error" in console.export_text()
    assert session.config.agent.model == "groq/my-model"
    session, console = build_session(tmp_path, ["q"])
    asyncio.run(session.submit("/model"))
    assert session.config.agent.model == "groq/llama"


def test_picker_blocks_providers_without_key(tmp_path, monkeypatch):
    only_keys(monkeypatch)
    session, console = build_session(tmp_path, ["openai", "q"])
    asyncio.run(session.submit("/model"))
    assert "set OPENAI_API_KEY and restart" in " ".join(console.export_text().split())
    assert session.config.agent.model == "groq/llama"


def test_direct_model_forms_still_work(tmp_path):
    session, console = build_session(tmp_path, [])
    asyncio.run(session.submit("/model openrouter/some/model"))
    assert session.config.agent.model == "openrouter/some/model"
    asyncio.run(session.submit("/model nope"))
    assert "Unknown provider" in console.export_text()


@pytest.mark.parametrize("width", [38, 40, 46, 80])
def test_banner_shows_at_phone_widths(tmp_path, width):
    session, console = build_session(tmp_path, [])
    console.width = width
    session.renderer.startup(session.config)
    output = console.export_text()
    assert "EURUS" in output and "A U T O N O M O U S" in output
    assert all(len(line) <= width for line in output.splitlines())


def test_tiny_terminal_falls_back_to_small_mark(tmp_path):
    session, console = build_session(tmp_path, [])
    console.width = 30
    session.renderer.startup(session.config)
    output = console.export_text()
    assert "EURUS" in output

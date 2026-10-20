import asyncio
import json

import pytest

from eurus.catalog import BUILTIN_PROVIDERS
from eurus.config import load_config
from eurus.errors import (
    AllProvidersFailedError,
    ConfigError,
    PolicyError,
    ProviderUnavailableError,
)
from eurus.models import Message, ModelRef
from eurus.providers.base import RetryPolicy
from eurus.providers.factory import build_router, resolve_chain
from eurus.providers.openai_compat import OpenAICompatProvider
from eurus.providers.router import Router
from eurus.providers.testing import FakeTransport, RecordingSleep
from eurus.providers.transport import HttpError, TransportError

MESSAGES = [Message.user("hi")]
OK = {"choices": [{"message": {"content": "from-b"}, "finish_reason": "stop"}]}


def provider(name, transport):
    return OpenAICompatProvider(
        BUILTIN_PROVIDERS[name], "k", transport, RetryPolicy(max_retries=0), RecordingSleep()
    )


def router_for(a_transport, b_transport):
    return Router(
        {"groq": provider("groq", a_transport), "openai": provider("openai", b_transport)},
        [ModelRef("groq", "a"), ModelRef("openai", "b")],
    )


def test_chat_falls_back_to_next_model():
    a = FakeTransport(json_responses=[HttpError(500, "down")])
    b = FakeTransport(json_responses=[OK])
    router = router_for(a, b)
    response = asyncio.run(router.chat(MESSAGES))
    assert response.message.content == "from-b"
    assert router.last_used == ModelRef("openai", "b")
    assert a.calls[0].payload["model"] == "a"
    assert b.calls[0].payload["model"] == "b"


def test_all_models_failing_is_reported():
    a = FakeTransport(json_responses=[HttpError(500, "down")])
    b = FakeTransport(json_responses=[HttpError(401, "bad key")])
    with pytest.raises(AllProvidersFailedError) as info:
        asyncio.run(router_for(a, b).chat(MESSAGES))
    assert len(info.value.failures) == 2


def test_single_failure_raises_the_original_error():
    a = FakeTransport(json_responses=[HttpError(500, "down")])
    router = Router({"groq": provider("groq", a)}, [ModelRef("groq", "a")])
    with pytest.raises(ProviderUnavailableError):
        asyncio.run(router.chat(MESSAGES))


def test_stream_falls_back_before_first_event():
    a = FakeTransport(streams=[HttpError(503, "busy")])
    chunk = json.dumps({"choices": [{"delta": {"content": "b-text"}}]})
    b = FakeTransport(streams=[[chunk, "[DONE]"]])
    router = router_for(a, b)

    async def go():
        return [event async for event in router.stream(MESSAGES)]

    events = asyncio.run(go())
    assert events[0].text == "b-text"
    assert router.last_used == ModelRef("openai", "b")


def test_stream_does_not_fall_back_after_first_event():
    chunk = json.dumps({"choices": [{"delta": {"content": "partial"}}]})
    a = FakeTransport(streams=[[chunk, TransportError("boom")]])
    b = FakeTransport(streams=[[chunk, "[DONE]"]])
    router = router_for(a, b)

    async def go():
        return [event async for event in router.stream(MESSAGES)]

    with pytest.raises(ProviderUnavailableError):
        asyncio.run(go())
    assert b.calls == []


def config_for(tmp_path, text="", **env):
    path = tmp_path / "c.toml"
    path.write_text(text, encoding="utf-8")
    return load_config(config_path=path, environ={"EURUS_HOME": str(tmp_path), **env})


def test_resolve_chain_dedupes_and_validates(tmp_path):
    config = config_for(tmp_path, '[agent]\nmodel = "groq/a"\nfallback = ["groq/a", "openai/b"]\n')
    assert resolve_chain(config) == [ModelRef("groq", "a"), ModelRef("openai", "b")]
    assert resolve_chain(config, "gemini/c")[0] == ModelRef("gemini", "c")


def test_resolve_chain_requires_a_model(tmp_path):
    with pytest.raises(ConfigError):
        resolve_chain(config_for(tmp_path))


def test_resolve_chain_rejects_unknown_provider(tmp_path):
    with pytest.raises(ConfigError):
        resolve_chain(config_for(tmp_path), "nope/x")


def test_local_only_filters_remote_models(tmp_path):
    text = '[agent]\nmodel = "groq/a"\nfallback = ["ollama/llama"]\n[network]\nlocal_only = true\n'
    config = config_for(tmp_path, text)
    assert resolve_chain(config) == [ModelRef("ollama", "llama")]


def test_local_only_blocks_when_nothing_is_local(tmp_path):
    text = '[agent]\nmodel = "groq/a"\n[network]\nlocal_only = true\n'
    with pytest.raises(PolicyError):
        resolve_chain(config_for(tmp_path, text))


def test_build_router_uses_environment_keys(tmp_path):
    config = config_for(tmp_path, '[agent]\nmodel = "groq/a"\n')
    transport = FakeTransport(json_responses=[OK])
    router = build_router(config, transport=transport, environ={"GROQ_API_KEY": "secret-k"})
    asyncio.run(router.chat(MESSAGES))
    assert transport.calls[0].headers["Authorization"] == "Bearer secret-k"
    assert transport.calls[0].payload["temperature"] == 0.2

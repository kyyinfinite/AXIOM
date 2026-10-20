from __future__ import annotations

import json

import httpx
import pytest

from eurus.catalog import ProviderSpec
from eurus.models import Message
from eurus.providers.anthropic import AnthropicProvider
from eurus.providers.base import ChatRequest, ProviderError
from eurus.providers.gemini import GeminiProvider
from eurus.providers.httpx_transport import HttpxTransport
from eurus.providers.openai_compat import OpenAICompatProvider, _finish, _usage
from eurus.providers.registry import create_provider


def test_openai_finish_and_usage_normalization() -> None:
    assert _finish("tool_calls", True) == "tool_calls"
    assert _finish("stop", False) == "stop"
    assert _usage({"prompt_tokens": 7, "completion_tokens": 3}).total_tokens == 10


def test_create_provider_by_protocol() -> None:
    local = ProviderSpec("local", "openai", "http://localhost:1234/v1")
    assert isinstance(create_provider(local), OpenAICompatProvider)
    claude = ProviderSpec("claude", "anthropic", "https://api.anthropic.com")
    assert isinstance(create_provider(claude), AnthropicProvider)
    gemini = ProviderSpec("gemini", "gemini", "https://example.test")
    assert isinstance(create_provider(gemini), GeminiProvider)


@pytest.mark.asyncio
async def test_openai_provider_posts_expected_payload() -> None:
    seen: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        body = {
            "model": "mock",
            "choices": [{"message": {"content": "pong"}, "finish_reason": "stop"}],
        }
        return httpx.Response(200, json=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = HttpxTransport(client=client)
    spec = ProviderSpec("mock", "openai", "https://mock.test/v1")
    provider = OpenAICompatProvider(spec, "test-key", transport)
    response = await provider.chat(ChatRequest("model-x", (Message.user("ping"),)))
    await client.aclose()
    assert response.message.content == "pong"
    assert seen["url"] == "https://mock.test/v1/chat/completions"
    assert seen["payload"]["messages"][0]["content"] == "ping"  # type: ignore[index]


@pytest.mark.asyncio
async def test_provider_classifies_retryable_http_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = HttpxTransport(client=client)
    provider = OpenAICompatProvider(
        ProviderSpec("mock", "openai", "https://mock.test/v1"), "test-key", transport
    )
    with pytest.raises(ProviderError) as caught:
        await provider.chat(ChatRequest("model-x", (Message.user("ping"),)))
    await client.aclose()
    assert caught.value.retryable is True
    assert caught.value.status == 429

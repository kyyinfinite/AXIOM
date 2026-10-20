import asyncio

import httpx
import pytest

from eurus.providers.httpx_transport import HttpxTransport
from eurus.providers.transport import HttpError, TransportError


def transport_for(handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return HttpxTransport(client=client)


def test_post_json_success():
    def handler(request):
        assert request.headers["authorization"] == "Bearer k"
        return httpx.Response(200, json={"ok": True})

    transport = transport_for(handler)
    result = asyncio.run(transport.post_json("https://x.test/a", {"authorization": "Bearer k"}, {}))
    assert result == {"ok": True}


def test_post_json_http_error_carries_retry_after():
    def handler(request):
        return httpx.Response(429, json={"error": "slow"}, headers={"retry-after": "2"})

    transport = transport_for(handler)
    with pytest.raises(HttpError) as info:
        asyncio.run(transport.post_json("https://x.test/a", {}, {}))
    assert info.value.status == 429
    assert info.value.retry_after == 2.0


def test_post_json_timeout_becomes_transport_error():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    transport = transport_for(handler)
    with pytest.raises(TransportError):
        asyncio.run(transport.post_json("https://x.test/a", {}, {}))


def test_stream_sse_yields_data_lines_only():
    body = b'event: message\ndata: {"a": 1}\n\n: ping\ndata: [DONE]\n\n'

    def handler(request):
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    transport = transport_for(handler)

    async def go():
        return [data async for data in transport.stream_sse("https://x.test/s", {}, {})]

    assert asyncio.run(go()) == ['{"a": 1}', "[DONE]"]


def test_stream_sse_http_error():
    def handler(request):
        return httpx.Response(500, text="down")

    transport = transport_for(handler)

    async def go():
        return [data async for data in transport.stream_sse("https://x.test/s", {}, {})]

    with pytest.raises(HttpError) as info:
        asyncio.run(go())
    assert info.value.status == 500
    assert info.value.body == "down"

import asyncio

import pytest

httpx = pytest.importorskip("httpx")

from eurus.providers.httpx_transport import HttpxTransport  # noqa: E402
from eurus.providers.transport import HttpError, TransportError  # noqa: E402


def transport_for(handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return HttpxTransport(client=client)


def blocked():
    return httpx.Response(403, text="error code: 1010")


def test_default_user_agent_is_not_the_python_one():
    seen = []

    def handler(request):
        seen.append(request.headers["user-agent"])
        return httpx.Response(200, json={"ok": True})

    asyncio.run(transport_for(handler).post_json("https://x.test/a", {}, {}))
    assert seen[0].startswith("eurus/")


def test_cloudflare_block_retries_once_with_a_second_user_agent():
    seen = []

    def handler(request):
        seen.append(request.headers["user-agent"])
        return blocked() if len(seen) == 1 else httpx.Response(200, json={"ok": True})

    result = asyncio.run(transport_for(handler).post_json("https://x.test/a", {}, {}))
    assert result == {"ok": True}
    assert len(seen) == 2
    assert seen[1].startswith("Mozilla/5.0 (compatible; eurus/")


def test_custom_user_agent_is_respected_and_not_retried():
    seen = []

    def handler(request):
        seen.append(request.headers["user-agent"])
        return blocked()

    with pytest.raises(HttpError) as info:
        asyncio.run(
            transport_for(handler).post_json("https://x.test/a", {"User-Agent": "mine"}, {})
        )
    assert seen == ["mine"]
    assert info.value.status == 403


def test_json_403_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(403, json={"error": {"message": "no"}})

    with pytest.raises(HttpError):
        asyncio.run(transport_for(handler).post_json("https://x.test/a", {}, {}))
    assert len(calls) == 1


def collect_stream(transport):
    async def go():
        return [data async for data in transport.stream_sse("https://x.test/s", {}, {})]

    return asyncio.run(go())


def test_stream_retries_after_cloudflare_block():
    calls = []

    def handler(request):
        calls.append(request.headers["accept"])
        if len(calls) == 1:
            return blocked()
        return httpx.Response(200, text='data: {"a": 1}\n\ndata: [DONE]\n\n')

    assert collect_stream(transport_for(handler)) == ['{"a": 1}', "[DONE]"]
    assert calls == ["text/event-stream", "text/event-stream"]


def test_stream_with_non_sse_body_is_an_error_not_silence():
    def handler(request):
        return httpx.Response(200, text="<html>Please log in</html>")

    with pytest.raises(TransportError) as info:
        collect_stream(transport_for(handler))
    assert "Please log in" in str(info.value)
    assert "<html>" not in str(info.value)


def test_illegal_header_value_does_not_leak_the_key():
    def handler(request):
        return httpx.Response(200, json={})

    transport = transport_for(handler)
    with pytest.raises(TransportError) as info:
        headers = {"Authorization": "Bearer bad\nkey"}
        asyncio.run(transport.post_json("https://x.test/a", headers, {}))
    assert "bad" not in str(info.value)

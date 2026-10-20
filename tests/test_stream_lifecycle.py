import asyncio
import json

import pytest

from eurus.catalog import BUILTIN_PROVIDERS
from eurus.models import Message, ModelRef
from eurus.providers.base import RetryPolicy
from eurus.providers.openai_compat import OpenAICompatProvider
from eurus.providers.router import Router
from eurus.providers.testing import FakeTransport, RecordingSleep
from eurus.providers.transport import TransportError

MESSAGES = [Message.user("hi")]


def chunk(text):
    return json.dumps({"choices": [{"delta": {"content": text}}]})


class TrackingTransport(FakeTransport):
    """Transport yang mencatat kapan generator stream_sse benar-benar ditutup."""

    def __init__(self, script, hang=False):
        super().__init__()
        self.script = script
        self.hang = hang
        self.closed_streams = 0

    async def stream_sse(self, url, headers, payload):
        try:
            for entry in self.script:
                if isinstance(entry, Exception):
                    raise entry
                yield entry
            if self.hang:
                await asyncio.Event().wait()
        finally:
            self.closed_streams += 1


def make_router(transport):
    provider = OpenAICompatProvider(
        BUILTIN_PROVIDERS["groq"], "k", transport, RetryPolicy(max_retries=0), RecordingSleep()
    )
    return Router({"groq": provider}, [ModelRef("groq", "a")])


async def test_early_stop_closes_provider_stream_immediately():
    transport = TrackingTransport([chunk("a"), chunk("b"), "[DONE]"])
    events = make_router(transport).stream(MESSAGES)
    first = await events.__anext__()
    assert first.kind == "text"
    await events.aclose()
    # harus sudah tertutup sekarang, bukan menunggu finalizer event loop
    assert transport.closed_streams == 1


async def test_normal_completion_closes_provider_stream():
    transport = TrackingTransport([chunk("a"), "[DONE]"])
    collected = [event async for event in make_router(transport).stream(MESSAGES)]
    assert any(event.kind == "text" for event in collected)
    assert transport.closed_streams == 1


async def test_failure_after_first_event_closes_stream_and_raises():
    transport = TrackingTransport([chunk("a"), TransportError("boom")])
    router = make_router(transport)
    with pytest.raises(Exception, match="boom"):
        async for _ in router.stream(MESSAGES):
            pass
    assert transport.closed_streams == 1


async def test_failure_before_first_event_closes_stream_and_raises():
    transport = TrackingTransport([TransportError("down")])
    router = make_router(transport)
    with pytest.raises(Exception, match="down"):
        async for _ in router.stream(MESSAGES):
            pass
    assert transport.closed_streams == 1


async def test_cancellation_closes_provider_stream():
    transport = TrackingTransport([chunk("a")], hang=True)
    router = make_router(transport)
    got_first = asyncio.Event()

    async def consume():
        async for _ in router.stream(MESSAGES):
            got_first.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(got_first.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert transport.closed_streams == 1

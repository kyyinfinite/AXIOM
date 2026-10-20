import asyncio
import json

import pytest

from eurus.catalog import BUILTIN_PROVIDERS
from eurus.errors import ProviderUnavailableError
from eurus.models import Message, ToolCall
from eurus.providers.anthropic import AnthropicProvider
from eurus.providers.base import ChatRequest, RetryPolicy
from eurus.providers.testing import FakeTransport, RecordingSleep

REQUEST = ChatRequest(model="claude-x", messages=(Message.system("be brief"), Message.user("hi")))


def make(transport, retries=0):
    return AnthropicProvider(
        BUILTIN_PROVIDERS["anthropic"],
        "k",
        transport,
        RetryPolicy(max_retries=retries),
        RecordingSleep(),
    )


def events_of(provider, request=REQUEST):
    async def go():
        return [event async for event in provider.stream(request)]

    return asyncio.run(go())


def sse(*objects):
    return [json.dumps(obj) for obj in objects]


def test_request_shape_and_system_extraction():
    body = {
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 4, "output_tokens": 1},
    }
    transport = FakeTransport(json_responses=[body])
    response = asyncio.run(make(transport).chat(REQUEST))
    call = transport.calls[0]
    assert call.url == "https://api.anthropic.com/v1/messages"
    assert call.headers["x-api-key"] == "k"
    assert call.headers["anthropic-version"] == "2023-06-01"
    assert call.payload["system"] == "be brief"
    assert call.payload["max_tokens"] == 4096
    assert call.payload["messages"] == [{"role": "user", "content": "hi"}]
    assert response.message.content == "hello"
    assert response.usage.total_tokens == 5
    assert response.finish_reason == "stop"


def test_tool_use_response():
    body = {
        "content": [
            {"type": "text", "text": "running"},
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}},
        ],
        "stop_reason": "tool_use",
        "usage": {},
    }
    response = asyncio.run(make(FakeTransport(json_responses=[body])).chat(REQUEST))
    assert response.finish_reason == "tool_calls"
    assert response.message.content == "running"
    assert response.message.tool_calls == (ToolCall("t1", "bash", {"cmd": "ls"}),)


def test_tool_results_are_merged_into_one_user_message():
    transport = FakeTransport(json_responses=[{"content": [], "usage": {}}])
    history = (
        Message.user("go"),
        Message.assistant("", (ToolCall("a", "bash", {}), ToolCall("b", "bash", {}))),
        Message.tool("a", "one"),
        Message.tool("b", "two"),
    )
    asyncio.run(make(transport).chat(ChatRequest(model="m", messages=history)))
    messages = transport.calls[0].payload["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert [b["tool_use_id"] for b in messages[2]["content"]] == ["a", "b"]
    assert messages[1]["content"][0]["type"] == "tool_use"


def test_stream_text_and_tool_use():
    stream = sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 7, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "t1", "name": "bash"},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"cmd"'},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": ': "ls"}'},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 9},
        },
        {"type": "message_stop"},
    )
    events = events_of(make(FakeTransport(streams=[stream])))
    assert [e.kind for e in events] == ["text", "tool_call", "usage", "done"]
    assert events[0].text == "Hi"
    assert events[1].tool_call == ToolCall("t1", "bash", {"cmd": "ls"})
    assert events[2].usage.input_tokens == 7
    assert events[2].usage.output_tokens == 9
    assert events[3].finish_reason == "tool_calls"


def test_stream_error_event_is_unavailable():
    stream = sse({"type": "error", "error": {"type": "overloaded_error", "message": "busy"}})
    with pytest.raises(ProviderUnavailableError):
        events_of(make(FakeTransport(streams=[stream])))


def test_stable_system_tier_uses_anthropic_cache_control():
    messages = (
        Message.system("<eurus-tier:stable>\nStable instructions\n</eurus-tier:stable>"),
        Message.system("<eurus-tier:context>\nProject context\n</eurus-tier:context>"),
        Message.user("hi"),
    )
    transport = FakeTransport(json_responses=[{"content": [], "usage": {}}])
    asyncio.run(make(transport).chat(ChatRequest(model="m", messages=messages)))
    system = transport.calls[0].payload["system"]
    assert system[0]["text"] == "Stable instructions"
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert system[1]["text"] == "Project context"


def test_cache_usage_metrics_are_preserved():
    body = {
        "content": [],
        "usage": {
            "input_tokens": 12,
            "output_tokens": 3,
            "cache_creation_input_tokens": 8,
            "cache_read_input_tokens": 5,
        },
    }
    response = asyncio.run(make(FakeTransport(json_responses=[body])).chat(REQUEST))
    assert response.usage.cache_creation_tokens == 8
    assert response.usage.cache_read_tokens == 5

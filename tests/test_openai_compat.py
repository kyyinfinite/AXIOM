import asyncio
import json

import pytest

from eurus.catalog import BUILTIN_PROVIDERS
from eurus.errors import (
    AuthenticationError,
    ProviderUnavailableError,
)
from eurus.models import Message, ToolCall
from eurus.providers.base import ChatRequest, RetryPolicy, ToolSpec, collect
from eurus.providers.openai_compat import OpenAICompatProvider
from eurus.providers.testing import FakeTransport, RecordingSleep
from eurus.providers.transport import HttpError, TransportError

REQUEST = ChatRequest(model="m", messages=(Message.user("hi"),))


def make(transport, name="groq", key="k", retries=2):
    sleep = RecordingSleep()
    provider = OpenAICompatProvider(
        BUILTIN_PROVIDERS[name], key, transport, RetryPolicy(max_retries=retries), sleep
    )
    return provider, sleep


def events_of(provider, request=REQUEST):
    async def go():
        return [event async for event in provider.stream(request)]

    return asyncio.run(go())


def chunk(**delta):
    return json.dumps({"choices": [{"delta": delta}]})


OK_BODY = {
    "model": "m",
    "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
}


def test_chat_text_and_request_shape():
    transport = FakeTransport(json_responses=[OK_BODY])
    provider, _ = make(transport)
    response = asyncio.run(provider.chat(REQUEST))
    assert response.message.content == "hello"
    assert response.usage.total_tokens == 5
    assert response.finish_reason == "stop"
    assert response.provider == "groq"
    call = transport.calls[0]
    assert call.url == "https://api.groq.com/openai/v1/chat/completions"
    assert call.headers["Authorization"] == "Bearer k"
    assert call.payload["model"] == "m"
    assert call.payload["messages"] == [{"role": "user", "content": "hi"}]
    assert "stream" not in call.payload


def test_chat_tool_call_response():
    body = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    provider, _ = make(FakeTransport(json_responses=[body]))
    response = asyncio.run(provider.chat(REQUEST))
    assert response.finish_reason == "tool_calls"
    assert response.message.tool_calls == (ToolCall("c1", "bash", {"cmd": "ls"}),)


def test_invalid_tool_arguments_are_preserved():
    body = {
        "choices": [
            {
                "message": {
                    "tool_calls": [{"function": {"name": "bash", "arguments": "not json"}}]
                },
                "finish_reason": "stop",
            }
        ]
    }
    provider, _ = make(FakeTransport(json_responses=[body]))
    response = asyncio.run(provider.chat(REQUEST))
    call = response.message.tool_calls[0]
    assert call.id == "call_0"
    assert call.arguments == {"_invalid_arguments": "not json"}
    assert response.finish_reason == "tool_calls"


def test_wire_format_for_tools_and_history():
    transport = FakeTransport(json_responses=[OK_BODY])
    provider, _ = make(transport)
    history = (
        Message.user("list files"),
        Message.assistant("", (ToolCall("c1", "bash", {"cmd": "ls"}),)),
        Message.tool("c1", "a.txt", "bash"),
    )
    request = ChatRequest(
        model="m",
        messages=history,
        tools=(ToolSpec("bash", "run a command", {"type": "object", "properties": {}}),),
        temperature=0.1,
        max_tokens=50,
    )
    asyncio.run(provider.chat(request))
    payload = transport.calls[0].payload
    assistant = payload["messages"][1]
    assert assistant["content"] is None
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"cmd": "ls"}'
    assert payload["messages"][2] == {"role": "tool", "tool_call_id": "c1", "content": "a.txt"}
    assert payload["tools"][0]["function"]["name"] == "bash"
    assert payload["temperature"] == 0.1
    assert payload["max_tokens"] == 50


def test_missing_key_raises_before_any_request():
    transport = FakeTransport()
    provider, _ = make(transport, key=None)
    with pytest.raises(AuthenticationError):
        asyncio.run(provider.chat(REQUEST))
    assert transport.calls == []


def test_keyless_provider_sends_no_auth_header():
    transport = FakeTransport(json_responses=[OK_BODY])
    provider, _ = make(transport, name="ollama", key=None)
    asyncio.run(provider.chat(REQUEST))
    assert "Authorization" not in transport.calls[0].headers


def test_rate_limit_is_retried_with_retry_after():
    transport = FakeTransport(json_responses=[HttpError(429, "{}", 1.5), OK_BODY])
    provider, sleep = make(transport)
    response = asyncio.run(provider.chat(REQUEST))
    assert response.message.content == "hello"
    assert len(transport.calls) == 2
    assert sleep.delays == [1.5]


def test_auth_error_is_not_retried():
    transport = FakeTransport(json_responses=[HttpError(401, '{"error": {"message": "nope"}}')])
    provider, sleep = make(transport)
    with pytest.raises(AuthenticationError):
        asyncio.run(provider.chat(REQUEST))
    assert len(transport.calls) == 1
    assert sleep.delays == []


def test_retries_are_exhausted():
    transport = FakeTransport(json_responses=[HttpError(500, "x")] * 3)
    provider, sleep = make(transport)
    with pytest.raises(ProviderUnavailableError):
        asyncio.run(provider.chat(REQUEST))
    assert len(transport.calls) == 3
    assert sleep.delays == [0.5, 1.0]


def test_network_error_is_retryable():
    transport = FakeTransport(json_responses=[TransportError("timed out"), OK_BODY])
    provider, sleep = make(transport)
    assert asyncio.run(provider.chat(REQUEST)).message.content == "hello"
    assert sleep.delays == [0.5]


def test_stream_text_usage_and_done():
    transport = FakeTransport(
        streams=[
            [
                chunk(content="he"),
                json.dumps({"choices": [{"delta": {"content": "llo"}, "finish_reason": "stop"}]}),
                json.dumps({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}),
                "[DONE]",
            ]
        ]
    )
    provider, _ = make(transport)
    events = events_of(provider)
    assert [e.kind for e in events] == ["text", "text", "usage", "done"]
    assert "".join(e.text for e in events if e.kind == "text") == "hello"
    assert events[2].usage.total_tokens == 5
    assert events[3].finish_reason == "stop"
    payload = transport.calls[0].payload
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


def test_stream_tool_call_accumulation():
    first = chunk(
        tool_calls=[{"index": 0, "id": "c1", "function": {"name": "bash", "arguments": '{"cmd"'}}]
    )
    second = chunk(tool_calls=[{"index": 0, "function": {"arguments": ': "ls"}'}}])
    finish = json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    provider, _ = make(FakeTransport(streams=[[first, second, finish, "[DONE]"]]))
    events = events_of(provider)
    assert [e.kind for e in events] == ["tool_call", "done"]
    assert events[0].tool_call == ToolCall("c1", "bash", {"cmd": "ls"})
    assert events[1].finish_reason == "tool_calls"


def test_stream_failure_before_first_event_is_retried():
    transport = FakeTransport(
        streams=[HttpError(503, "busy"), [chunk(content="ok"), "[DONE]"]],
    )
    provider, sleep = make(transport)
    events = events_of(provider)
    assert events[0].text == "ok"
    assert sleep.delays == [0.5]


def test_stream_failure_after_first_event_is_not_retried():
    transport = FakeTransport(streams=[[chunk(content="partial"), TransportError("boom")]])
    provider, sleep = make(transport)
    seen = []

    async def go():
        async for event in provider.stream(REQUEST):
            seen.append(event)

    with pytest.raises(ProviderUnavailableError):
        asyncio.run(go())
    assert [e.text for e in seen] == ["partial"]
    assert sleep.delays == []


def test_collect_builds_a_response():
    provider, _ = make(
        FakeTransport(streams=[[chunk(content="he"), chunk(content="llo"), "[DONE]"]])
    )
    response = asyncio.run(collect(provider.stream(REQUEST), provider="groq", model="m"))
    assert response.message.content == "hello"
    assert response.finish_reason == "stop"

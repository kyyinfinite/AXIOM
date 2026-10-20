import asyncio
import json

from eurus.catalog import BUILTIN_PROVIDERS
from eurus.models import Message, ToolCall
from eurus.providers.base import ChatRequest, RetryPolicy, ToolSpec
from eurus.providers.gemini import GeminiProvider, clean_schema
from eurus.providers.testing import FakeTransport, RecordingSleep

REQUEST = ChatRequest(model="gemini-x", messages=(Message.system("brief"), Message.user("hi")))


def make(transport):
    return GeminiProvider(
        BUILTIN_PROVIDERS["gemini"], "k", transport, RetryPolicy(max_retries=0), RecordingSleep()
    )


def events_of(provider, request=REQUEST):
    async def go():
        return [event async for event in provider.stream(request)]

    return asyncio.run(go())


def test_request_shape_and_text_response():
    body = {
        "candidates": [{"content": {"parts": [{"text": "hello"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2},
    }
    transport = FakeTransport(json_responses=[body])
    request = ChatRequest(
        model="models/gemini-x",
        messages=REQUEST.messages,
        temperature=0.3,
        max_tokens=100,
    )
    response = asyncio.run(make(transport).chat(request))
    call = transport.calls[0]
    assert call.url == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-x:generateContent"
    )
    assert call.headers["x-goog-api-key"] == "k"
    assert call.payload["systemInstruction"] == {"parts": [{"text": "brief"}]}
    assert call.payload["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert call.payload["generationConfig"] == {"temperature": 0.3, "maxOutputTokens": 100}
    assert response.message.content == "hello"
    assert response.usage.total_tokens == 5
    assert response.finish_reason == "stop"


def test_clean_schema_drops_unsupported_keys():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "$schema": "x",
        "properties": {
            "path": {"type": "string", "default": "a", "title": "Path"},
            "items": {"type": "array", "items": {"type": "string", "additionalProperties": True}},
        },
        "required": ["path"],
    }
    assert clean_schema(schema) == {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "items": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["path"],
    }


def test_tools_wire_and_empty_parameters_omitted():
    transport = FakeTransport(json_responses=[{"candidates": []}])
    tools = (
        ToolSpec(
            "read",
            "read a file",
            {"type": "object", "properties": {"p": {"type": "string"}}},
        ),
        ToolSpec("now", "current time"),
    )
    asyncio.run(
        make(transport).chat(ChatRequest(model="m", messages=REQUEST.messages, tools=tools))
    )
    declarations = transport.calls[0].payload["tools"][0]["functionDeclarations"]
    assert "parameters" in declarations[0]
    assert "parameters" not in declarations[1]


def test_function_call_roundtrip_keeps_signature():
    body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {"name": "bash", "args": {"cmd": "ls"}},
                            "thoughtSignature": "sig",
                        }
                    ]
                },
                "finishReason": "STOP",
            }
        ]
    }
    transport = FakeTransport(json_responses=[body, {"candidates": []}])
    provider = make(transport)
    response = asyncio.run(provider.chat(REQUEST))
    call = response.message.tool_calls[0]
    assert response.finish_reason == "tool_calls"
    assert call.name == "bash"
    assert call.arguments == {"cmd": "ls"}
    assert call.extra == {"thoughtSignature": "sig"}

    history = (
        Message.user("go"),
        response.message,
        Message.tool(call.id, "a.txt"),
    )
    asyncio.run(provider.chat(ChatRequest(model="m", messages=history)))
    contents = transport.calls[1].payload["contents"]
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"][0]["thoughtSignature"] == "sig"
    reply = contents[2]["parts"][0]["functionResponse"]
    assert reply == {"name": "bash", "response": {"result": "a.txt"}}


def test_parallel_tool_responses_share_one_turn():
    transport = FakeTransport(json_responses=[{"candidates": []}])
    history = (
        Message.user("go"),
        Message.assistant("", (ToolCall("a", "x", {}), ToolCall("b", "y", {}))),
        Message.tool("a", "1"),
        Message.tool("b", "2"),
    )
    asyncio.run(make(transport).chat(ChatRequest(model="m", messages=history)))
    contents = transport.calls[0].payload["contents"]
    assert len(contents) == 3
    assert len(contents[2]["parts"]) == 2


def test_blocked_prompt_reports_content_filter():
    body = {"promptFeedback": {"blockReason": "SAFETY"}}
    response = asyncio.run(make(FakeTransport(json_responses=[body])).chat(REQUEST))
    assert response.finish_reason == "content_filter"
    assert response.message.content == ""


def test_stream_text_tool_and_usage():
    chunks = [
        json.dumps({"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]}),
        json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "lo"},
                                {"functionCall": {"name": "bash", "args": {}}},
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3},
            }
        ),
    ]
    transport = FakeTransport(streams=[chunks])
    events = events_of(make(transport))
    assert [e.kind for e in events] == ["text", "text", "tool_call", "usage", "done"]
    assert events[3].usage.total_tokens == 5
    assert events[4].finish_reason == "tool_calls"
    assert transport.calls[0].url.endswith(":streamGenerateContent?alt=sse")

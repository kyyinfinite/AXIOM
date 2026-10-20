import asyncio
import json
from types import SimpleNamespace

import pytest

from eurus.catalog import BUILTIN_PROVIDERS
from eurus.errors import AccessDeniedError, ConfigError, ProviderError, RateLimitError
from eurus.models import Message, ToolCall
from eurus.providers.base import ChatRequest, RetryPolicy, ToolSpec
from eurus.providers.factory import build_provider
from eurus.providers.genai_sdk import GenaiSdkProvider, decode_signatures, response_to_rest
from eurus.providers.groq_sdk import GroqSdkProvider
from eurus.providers.sdk import error_from_sdk, import_sdk
from eurus.providers.testing import FakeTransport, RecordingSleep

REQUEST = ChatRequest(model="m", messages=(Message.user("hi"),), max_tokens=5)


def run(coro):
    return asyncio.run(coro)


class Dumpable:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


class FakeApiError(Exception):
    def __init__(self, status_code, text="", headers=None):
        super().__init__("api error")
        self.status_code = status_code
        self.response = SimpleNamespace(text=text, headers=headers or {})


class AsyncChunks:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


def groq_client(result=None, error=None):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        if error is not None:
            raise error
        return result

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, calls


def groq_provider(client, retries=0):
    return GroqSdkProvider(
        BUILTIN_PROVIDERS["groq-sdk"],
        "k",
        FakeTransport(),
        RetryPolicy(max_retries=retries),
        RecordingSleep(),
        client=client,
    )


def test_groq_sdk_chat_reuses_openai_parsing():
    body = {
        "model": "m",
        "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }
    client, calls = groq_client(Dumpable(body))
    response = run(groq_provider(client).chat(REQUEST))
    assert response.message.content == "hello"
    assert response.usage.total_tokens == 5
    assert calls[0]["model"] == "m"
    assert calls[0]["max_tokens"] == 5
    assert "stream" not in calls[0]


def test_groq_sdk_stream_with_tool_call():
    chunks = [
        Dumpable({"choices": [{"delta": {"content": "x"}}]}),
        Dumpable(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "function": {"name": "t", "arguments": '{"a": 1}'},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        ),
        Dumpable(
            {"choices": [], "x_groq": {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}}
        ),
    ]
    client, calls = groq_client(AsyncChunks(chunks))

    async def go():
        return [event async for event in groq_provider(client).stream(REQUEST)]

    events = run(go())
    kinds = [event.kind for event in events]
    assert kinds == ["text", "tool_call", "usage", "done"]
    assert events[1].tool_call == ToolCall("c1", "t", {"a": 1})
    assert calls[0]["stream"] is True
    assert calls[0]["stream_options"] == {"include_usage": True}


def test_groq_sdk_maps_cloudflare_403_to_edge_blocked():
    client, _ = groq_client(error=FakeApiError(403, "error code: 1010"))
    with pytest.raises(AccessDeniedError) as info:
        run(groq_provider(client).chat(REQUEST))
    assert info.value.reason == "edge_blocked"


def test_groq_sdk_429_is_retried():
    error = FakeApiError(429, '{"error": {"message": "slow"}}', {"Retry-After": "2"})
    client, calls = groq_client(error=error)
    with pytest.raises(RateLimitError) as info:
        run(groq_provider(client, retries=2).chat(REQUEST))
    assert len(calls) == 3
    assert info.value.retry_after == 2.0


def test_error_from_sdk_without_status():
    class APIConnectionError(Exception):
        pass

    error = error_from_sdk("p", APIConnectionError("boom"))
    assert error.retryable is True
    assert isinstance(error_from_sdk("p", ValueError("x")), ProviderError)


def part(**kwargs):
    base = {"text": None, "thought": None, "function_call": None, "thought_signature": None}
    return SimpleNamespace(**{**base, **kwargs})


def genai_response(parts, reason="STOP"):
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=parts), finish_reason=SimpleNamespace(name=reason)
    )
    usage = SimpleNamespace(
        prompt_token_count=7, candidates_token_count=3, thoughts_token_count=None
    )
    return SimpleNamespace(
        candidates=[candidate], usage_metadata=usage, prompt_feedback=None, model_version="gm-1"
    )


def genai_provider(client):
    return GenaiSdkProvider(
        BUILTIN_PROVIDERS["gemini-sdk"],
        "k",
        FakeTransport(),
        RetryPolicy(max_retries=0),
        RecordingSleep(),
        client=client,
    )


def genai_client(response=None, chunks=(), error=None):
    calls = []

    async def generate_content(**kwargs):
        calls.append(kwargs)
        if error is not None:
            raise error
        return response

    async def generate_content_stream(**kwargs):
        calls.append(kwargs)
        return AsyncChunks(chunks)

    models = SimpleNamespace(
        generate_content=generate_content, generate_content_stream=generate_content_stream
    )
    return SimpleNamespace(aio=SimpleNamespace(models=models)), calls


def test_genai_chat_text_usage_and_request_shape():
    client, calls = genai_client(genai_response([part(text="hi there")]))
    tools = (ToolSpec("t", "d", {"type": "object", "properties": {"a": {"type": "string"}}}),)
    request = ChatRequest(
        model="gemini-x",
        messages=(Message.system("sys"), Message.user("hi")),
        tools=tools,
        temperature=0.1,
        max_tokens=9,
    )
    response = run(genai_provider(client).chat(request))
    assert response.message.content == "hi there"
    assert response.usage.input_tokens == 7
    assert response.usage.output_tokens == 3
    assert response.model == "gm-1"
    call = calls[0]
    assert call["model"] == "gemini-x"
    assert call["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    config = call["config"]
    assert config["system_instruction"] == "sys"
    assert config["max_output_tokens"] == 9
    assert config["automatic_function_calling"] == {"disable": True}
    assert config["tools"][0]["function_declarations"][0]["name"] == "t"


def test_genai_tool_call_keeps_thought_signature_roundtrip():
    call = SimpleNamespace(name="read", args={"path": "a.py"})
    raw_signature = b"\x01\x02sig"
    response = genai_response([part(function_call=call, thought_signature=raw_signature)])
    client, _ = genai_client(response)
    result = run(genai_provider(client).chat(REQUEST))
    tool_call = result.message.tool_calls[0]
    assert tool_call.name == "read"
    assert tool_call.arguments == {"path": "a.py"}
    assert result.finish_reason == "tool_calls"
    stored = tool_call.extra["thoughtSignature"]
    part_with_signature = {"functionCall": {"name": "read", "args": {}}, "thoughtSignature": stored}
    contents = [{"role": "model", "parts": [part_with_signature]}]
    assert decode_signatures(contents)[0]["parts"][0]["thoughtSignature"] == raw_signature


def test_genai_invalid_signature_is_dropped_instead_of_crashing():
    contents = [{"role": "model", "parts": [{"text": "x", "thoughtSignature": "%%%"}]}]
    assert decode_signatures(contents)[0]["parts"] == [{"text": "x"}]


def test_genai_stream_and_thoughts_are_hidden():
    chunks = [
        genai_response([part(text="secret", thought=True), part(text="a")], reason=None),
        genai_response([part(text="b")]),
    ]
    client, _ = genai_client(chunks=chunks)

    async def go():
        return [event async for event in genai_provider(client).stream(REQUEST)]

    events = run(go())
    assert "".join(e.text for e in events if e.kind == "text") == "ab"
    assert events[-1].kind == "done"


def test_genai_error_mapping():
    client, _ = genai_client(error=FakeApiError(403, '{"error": {"message": "API key invalid"}}'))
    with pytest.raises(AccessDeniedError):
        run(genai_provider(client).chat(REQUEST))


def test_response_to_rest_reports_block_reason():
    response = SimpleNamespace(
        candidates=[],
        usage_metadata=None,
        prompt_feedback=SimpleNamespace(block_reason=SimpleNamespace(name="SAFETY")),
        model_version=None,
    )
    assert response_to_rest(response)["promptFeedback"] == {"blockReason": "SAFETY"}
    client, _ = genai_client(response)
    assert run(genai_provider(client).chat(REQUEST)).finish_reason == "content_filter"


def test_missing_sdk_gives_an_install_hint():
    with pytest.raises(ConfigError) as info:
        import_sdk("eurus_surely_missing_package", "groq")
    assert "eurus-agent[groq]" in str(info.value)


def test_build_provider_rejects_unknown_protocol_and_serialises_json():
    spec = BUILTIN_PROVIDERS["groq"]
    odd = type(spec)("odd", "nope", "https://x.test")
    with pytest.raises(ProviderError):
        build_provider(odd, None, FakeTransport())
    assert json.dumps(Dumpable({"a": 1}).model_dump()) == '{"a": 1}'

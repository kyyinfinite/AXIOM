import asyncio
import json

import pytest

from eurus.catalog import (
    BUILTIN_PROVIDERS,
    builtin_providers,
    clean_secret,
    is_ready,
    resolve_api_key,
)
from eurus.config import build_config
from eurus.errors import (
    AccessDeniedError,
    AuthenticationError,
    ConfigError,
    InvalidRequestError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
)
from eurus.models import Message, ModelRef
from eurus.providers.base import ChatRequest, RetryPolicy
from eurus.providers.openai_compat import OpenAICompatProvider
from eurus.providers.testing import FakeTransport, RecordingSleep
from eurus.providers.transport import (
    HttpError,
    error_from_http,
    error_from_payload,
    extract_error_message,
    is_cloudflare_block,
)

REQUEST = ChatRequest(model="m", messages=(Message.user("hi"),), max_tokens=10, temperature=0.2)
CLOUDFLARE_BODY = "error code: 1010"
CLOUDFLARE_HTML = "<html><head><title>Attention Required! | Cloudflare</title></head></html>"
OK_BODY = {
    "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
}


def make(spec_name="groq", transport=None, retries=0):
    return OpenAICompatProvider(
        BUILTIN_PROVIDERS[spec_name],
        "k",
        transport or FakeTransport(),
        RetryPolicy(max_retries=retries),
        RecordingSleep(),
    )


def run(coro):
    return asyncio.run(coro)


def events_of(provider, request=REQUEST):
    async def go():
        return [event async for event in provider.stream(request)]

    return run(go())


@pytest.mark.parametrize("body", [CLOUDFLARE_BODY, CLOUDFLARE_HTML])
def test_cloudflare_403_is_edge_blocked_not_a_key_error(body):
    error = error_from_http("groq", HttpError(403, body))
    assert isinstance(error, AccessDeniedError)
    assert isinstance(error, AuthenticationError)
    assert error.reason == "edge_blocked"
    assert error.retryable is False
    assert "Cloudflare" in str(error)
    assert "<html>" not in str(error)


def test_json_403_is_not_mistaken_for_cloudflare():
    body = json.dumps({"error": {"message": "nope", "type": "permissions_error"}})
    error = error_from_http("groq", HttpError(403, body, headers={"cf-ray": "abc"}))
    assert isinstance(error, AccessDeniedError)
    assert error.reason == "model_blocked"
    assert not is_cloudflare_block(HttpError(403, body, headers={"cf-ray": "abc"}))


def test_plain_403_and_401_are_distinguished():
    forbidden = error_from_http("x", HttpError(403, '{"error": "bad"}'))
    assert isinstance(forbidden, AccessDeniedError)
    assert forbidden.reason == "forbidden"
    unauthorized = error_from_http("x", HttpError(401, '{"error": "bad"}'))
    assert isinstance(unauthorized, AuthenticationError)
    assert not isinstance(unauthorized, AccessDeniedError)


def test_nim_account_404_gets_a_hint():
    body = json.dumps({"status": 404, "detail": "Function 'abc': Not found for account 'z'"})
    error = error_from_http("nvidia", HttpError(404, body))
    assert isinstance(error, InvalidRequestError)
    assert "not available to your account" in str(error)


def test_extract_error_message_handles_detail_and_html():
    assert extract_error_message('{"status": 401, "detail": "denied"}') == "denied"
    assert extract_error_message(CLOUDFLARE_HTML) == "Attention Required! | Cloudflare"


@pytest.mark.parametrize(
    ("payload", "kind", "retryable"),
    [
        ({"error": {"message": "slow down", "code": 429}}, RateLimitError, True),
        ({"error": {"message": "Rate limit reached"}}, RateLimitError, True),
        ({"error": {"message": "bad api key", "code": 401}}, AuthenticationError, False),
        ({"error": {"message": "overloaded"}}, ProviderUnavailableError, True),
        ({"error": {"message": "bad", "code": 400}}, InvalidRequestError, False),
        ({"error": "odd"}, ProviderError, False),
    ],
)
def test_error_from_payload(payload, kind, retryable):
    error = error_from_payload("p", payload)
    assert type(error) is kind
    assert error.retryable is retryable
    assert error_from_payload("p", {"choices": []}) is None


def test_chat_200_with_embedded_error_is_raised():
    transport = FakeTransport(json_responses=[{"error": {"message": "quota exceeded"}}])
    with pytest.raises(RateLimitError):
        run(make(transport=transport).chat(REQUEST))


def test_stream_embedded_error_is_raised():
    chunk = json.dumps({"error": {"message": "server_error", "code": 500}})
    transport = FakeTransport(streams=[[chunk]])
    with pytest.raises(ProviderUnavailableError):
        events_of(make(transport=transport))


def test_empty_stream_is_a_retryable_failure_not_silence():
    transport = FakeTransport(streams=[["[DONE]"]])
    with pytest.raises(ProviderUnavailableError) as info:
        events_of(make(transport=transport))
    assert info.value.retryable is True


def test_empty_stream_is_retried_then_succeeds():
    good = [json.dumps({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]})]
    transport = FakeTransport(streams=[["[DONE]"], good])
    events = events_of(make(transport=transport, retries=1))
    assert [e.text for e in events if e.kind == "text"] == ["hi"]


def test_groq_usage_extension_is_read():
    chunks = [
        json.dumps({"choices": [{"delta": {"content": "a"}}]}),
        json.dumps(
            {"choices": [], "x_groq": {"usage": {"prompt_tokens": 4, "completion_tokens": 2}}}
        ),
        "[DONE]",
    ]
    events = events_of(make(transport=FakeTransport(streams=[chunks])))
    usage = [e.usage for e in events if e.kind == "usage"]
    assert usage and usage[0].total_tokens == 6


def test_max_tokens_is_swapped_when_the_model_rejects_it():
    message = "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead."
    rejection = HttpError(400, json.dumps({"error": {"message": message}}))
    transport = FakeTransport(json_responses=[rejection, OK_BODY])
    response = run(make("openai", transport).chat(REQUEST))
    assert response.message.content == "hello"
    assert "max_tokens" in transport.calls[0].payload
    assert transport.calls[1].payload["max_completion_tokens"] == 10
    assert "max_tokens" not in transport.calls[1].payload


def test_temperature_is_dropped_when_the_model_rejects_it():
    message = "Unsupported value: 'temperature' does not support 0.2 with this model."
    rejection = HttpError(400, json.dumps({"error": {"message": message}}))
    transport = FakeTransport(json_responses=[rejection, OK_BODY])
    run(make("openai", transport).chat(REQUEST))
    assert "temperature" in transport.calls[0].payload
    assert "temperature" not in transport.calls[1].payload


def test_unrelated_400_is_not_swallowed():
    rejection = HttpError(400, json.dumps({"error": {"message": "context too long"}}))
    transport = FakeTransport(json_responses=[rejection])
    with pytest.raises(InvalidRequestError):
        run(make(transport=transport).chat(REQUEST))
    assert len(transport.calls) == 1


def test_nvidia_provider_defaults_and_payload():
    spec = BUILTIN_PROVIDERS["nvidia"]
    assert spec.base_url == "https://integrate.api.nvidia.com/v1"
    assert spec.api_key_env == "NVIDIA_API_KEY"
    transport = FakeTransport(json_responses=[OK_BODY], streams=[["[DONE]"]])
    provider = make("nvidia", transport)
    request = ChatRequest(
        model="meta/llama-3.3-70b-instruct",
        messages=(
            Message.system("one"),
            Message.user("hi"),
            Message.system("two"),
        ),
    )
    run(provider.chat(request))
    call = transport.calls[0]
    assert call.url == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert call.headers["Authorization"] == "Bearer k"
    roles = [message["role"] for message in call.payload["messages"]]
    assert roles == ["system", "user"]
    assert call.payload["messages"][0]["content"] == "one\n\ntwo"
    assert "stream_options" not in provider._payload(request, True)


def test_nvidia_model_ref_keeps_the_namespaced_model_id():
    ref = ModelRef.parse("nvidia/meta/llama-3.3-70b-instruct")
    assert ref.provider == "nvidia"
    assert ref.model == "meta/llama-3.3-70b-instruct"


def test_nvidia_base_url_override():
    providers = builtin_providers({"NVIDIA_BASE_URL": "http://localhost:8000/v1/"})
    assert providers["nvidia"].base_url == "http://localhost:8000/v1"


def test_provider_user_agent_and_headers_reach_the_request():
    raw = {
        "providers": {
            "groq": {"user_agent": "my-app/1.0", "headers": {"X-Team": "a"}},
        }
    }
    config = build_config(raw, {})
    spec = config.providers["groq"]
    assert spec.user_agent == "my-app/1.0"
    assert spec.extra_headers == (("X-Team", "a"),)
    headers = OpenAICompatProvider(spec, "k", FakeTransport())._headers()
    assert headers["User-Agent"] == "my-app/1.0"
    assert headers["X-Team"] == "a"


def test_config_rejects_header_injection():
    with pytest.raises(ConfigError):
        build_config({"providers": {"groq": {"user_agent": "a\nb"}}}, {})
    with pytest.raises(ConfigError):
        build_config({"providers": {"groq": {"headers": {"Bad Name": "x"}}}}, {})


def test_config_accepts_sdk_protocols():
    raw = {"providers": {"groq": {"protocol": "groq-sdk"}}}
    assert build_config(raw, {}).providers["groq"].protocol == "groq-sdk"
    assert build_config({}, {}).providers["gemini-sdk"].protocol == "genai-sdk"


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("  gsk_abc123  ", "gsk_abc123"),
        ('"gsk_abc123"', "gsk_abc123"),
        ("'nvapi-abc123'", "nvapi-abc123"),
        ("Bearer gsk_abc123", "gsk_abc123"),
        ("gsk_abc\r\n123", "gsk_abc123"),
        ("", ""),
    ],
)
def test_clean_secret(raw, clean):
    assert clean_secret(raw) == clean


def test_resolve_api_key_uses_cleaned_value():
    spec = BUILTIN_PROVIDERS["nvidia"]
    assert resolve_api_key(spec, {"NVIDIA_API_KEY": ' "nvapi-xyz" \r'}) == "nvapi-xyz"
    assert not is_ready(spec, {"NVIDIA_API_KEY": '""'})

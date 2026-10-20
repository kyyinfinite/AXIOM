import pytest

from eurus.errors import (
    AuthenticationError,
    InvalidRequestError,
    ProviderUnavailableError,
    RateLimitError,
)
from eurus.providers.base import RetryPolicy, parse_arguments
from eurus.providers.transport import (
    HttpError,
    error_from_http,
    extract_error_message,
    parse_sse_line,
)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("data: {}", "{}"),
        ("data:{}", "{}"),
        ("data:  x", " x"),
        ("event: message", None),
        (": keepalive", None),
        ("", None),
    ],
)
def test_parse_sse_line(line, expected):
    assert parse_sse_line(line) == expected


def test_extract_error_message_shapes():
    assert extract_error_message('{"error": {"message": "bad key"}}') == "bad key"
    assert extract_error_message('{"error": "plain"}') == "plain"
    assert extract_error_message('[{"error": {"message": "listed"}}]') == "listed"
    assert extract_error_message("not json") == "not json"


@pytest.mark.parametrize(
    ("status", "kind", "retryable"),
    [
        (401, AuthenticationError, False),
        (403, AuthenticationError, False),
        (429, RateLimitError, True),
        (500, ProviderUnavailableError, True),
        (529, ProviderUnavailableError, True),
        (400, InvalidRequestError, False),
        (404, InvalidRequestError, False),
    ],
)
def test_error_from_http(status, kind, retryable):
    error = error_from_http("groq", HttpError(status, "{}", 2.0))
    assert isinstance(error, kind)
    assert error.retryable is retryable
    assert error.status == status
    assert error.retry_after == 2.0


def test_retry_delay_grows_and_honours_retry_after():
    policy = RetryPolicy(max_retries=3, base_delay=0.5, max_delay=8.0, max_retry_after=30.0)
    assert policy.delay(0, None) == 0.5
    assert policy.delay(2, None) == 2.0
    assert policy.delay(10, None) == 8.0
    assert policy.delay(0, 5.0) == 5.0
    assert policy.delay(0, 500.0) == 30.0


def test_parse_arguments():
    assert parse_arguments('{"a": 1}') == {"a": 1}
    assert parse_arguments("") == {}
    assert parse_arguments(None) == {}
    assert parse_arguments({"a": 1}) == {"a": 1}
    assert parse_arguments("nope") == {"_invalid_arguments": "nope"}
    assert parse_arguments("[1]") == {"_invalid_arguments": "[1]"}

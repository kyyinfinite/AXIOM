from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Mapping
from typing import Any, Protocol

from eurus.errors import (
    AccessDeniedError,
    AuthenticationError,
    InvalidRequestError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
)

CLOUDFLARE_CODE = re.compile(r"error code:\s*1\d{3}", re.IGNORECASE)
MARKUP = re.compile(r"<[^>]+>")
MESSAGE_LIMIT = 300

EDGE_BLOCKED_HINT = (
    "the request was rejected by Cloudflare before it reached the API, so the key was never "
    "checked; set providers.<name>.user_agent or try the SDK protocol"
)
MODEL_BLOCKED_HINT = (
    "the key is valid but this model is blocked for your organization or project; "
    "enable it in the provider console"
)
REGION_BLOCKED_HINT = "the provider refused the request for your region or network"
FORBIDDEN_HINT = (
    "the key was rejected or lacks permission; regenerate it and check for quotes or spaces"
)
UNAUTHORIZED_HINT = "check that the API key is valid and was copied without quotes or spaces"
ACCOUNT_MODEL_HINT = "this model is not available to your account; pick another from the catalog"


class HttpError(Exception):
    def __init__(
        self,
        status: int,
        body: str,
        retry_after: float | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.body = body
        self.retry_after = retry_after
        self.headers = {key.lower(): value for key, value in (headers or {}).items()}


class TransportError(Exception):
    pass


class HttpTransport(Protocol):
    async def post_json(
        self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any]
    ) -> dict[str, Any]: ...

    def stream_sse(
        self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any]
    ) -> AsyncIterator[str]: ...

    async def aclose(self) -> None: ...


def parse_sse_line(line: str) -> str | None:
    if not line.startswith("data:"):
        return None
    value = line[5:]
    if value.startswith(" "):
        value = value[1:]
    return value


def strip_markup(text: str) -> str:
    return " ".join(MARKUP.sub(" ", text).split())


def _find_message(data: Any) -> str | None:
    if isinstance(data, list) and data:
        return _find_message(data[0])
    if not isinstance(data, dict):
        return None
    error = data.get("error")
    if isinstance(error, str):
        return error
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return str(error["message"])
    if isinstance(data.get("message"), str):
        return str(data["message"])
    if isinstance(data.get("detail"), str):
        return str(data["detail"])
    return None


def extract_error_message(body: str) -> str:
    text = body.strip()
    try:
        data = json.loads(text)
    except ValueError:
        return strip_markup(text)[:MESSAGE_LIMIT]
    return (_find_message(data) or text)[:MESSAGE_LIMIT]


def is_cloudflare_block(error: HttpError) -> bool:
    if error.status != 403:
        return False
    body = error.body.strip()
    if CLOUDFLARE_CODE.search(body):
        return True
    lowered = body.lower()
    if "cloudflare" in lowered or "attention required" in lowered:
        return True
    return "cf-ray" in error.headers and not body.startswith(("{", "["))


def classify_forbidden(error: HttpError, detail: str) -> tuple[str, str]:
    if is_cloudflare_block(error):
        return "edge_blocked", EDGE_BLOCKED_HINT
    lowered = f"{detail} {error.body}".lower()
    if "permissions_error" in lowered or (
        "model" in lowered
        and any(word in lowered for word in ("permission", "not allowed", "blocked", "enable"))
    ):
        return "model_blocked", MODEL_BLOCKED_HINT
    if any(word in lowered for word in ("region", "country", "territory", "location")):
        return "region_blocked", REGION_BLOCKED_HINT
    return "forbidden", FORBIDDEN_HINT


def error_from_http(provider: str, error: HttpError) -> ProviderError:
    detail = extract_error_message(error.body) or "no details"
    status = error.status
    message = f"{provider}: HTTP {status}: {detail}"

    def build(kind: type[ProviderError], retryable: bool = False) -> ProviderError:
        return kind(
            message,
            provider=provider,
            status=status,
            retryable=retryable,
            retry_after=error.retry_after,
        )

    if status == 403:
        reason, hint = classify_forbidden(error, detail)
        return AccessDeniedError(
            f"{message} [{reason}: {hint}]",
            reason=reason,
            provider=provider,
            status=status,
            retry_after=error.retry_after,
        )
    if status == 401:
        return AuthenticationError(
            f"{message} [{UNAUTHORIZED_HINT}]",
            provider=provider,
            status=status,
            retry_after=error.retry_after,
        )
    if status == 429:
        return build(RateLimitError, retryable=True)
    if status in (408, 409) or status >= 500:
        return build(ProviderUnavailableError, retryable=True)
    if status == 404 and "not found for account" in detail.lower():
        return InvalidRequestError(
            f"{message} [{ACCOUNT_MODEL_HINT}]",
            provider=provider,
            status=status,
            retry_after=error.retry_after,
        )
    if 400 <= status < 500:
        return build(InvalidRequestError)
    return build(ProviderError)


def error_from_payload(provider: str, data: Mapping[str, Any]) -> ProviderError | None:
    raw = data.get("error")
    if not raw:
        return None
    code: Any = None
    if isinstance(raw, dict):
        message = str(raw.get("message") or raw)
        code = raw.get("code")
        kind = str(raw.get("type") or raw.get("status") or code or "")
    else:
        message = str(raw)
        kind = ""
    status = code if isinstance(code, int) and not isinstance(code, bool) else None
    text = f"{kind} {message}".lower()
    full = f"{provider}: {message[:MESSAGE_LIMIT]}"
    options: dict[str, Any] = {"provider": provider, "status": status}
    if status == 429 or "rate limit" in text or "rate_limit" in text or "quota" in text:
        return RateLimitError(full, retryable=True, **options)
    if status in (401, 403) or any(
        word in text for word in ("unauthorized", "api key", "api_key", "permission")
    ):
        return AuthenticationError(full, **options)
    if (status is not None and status >= 500) or any(
        word in text for word in ("overload", "unavailable", "timeout", "server_error", "internal")
    ):
        return ProviderUnavailableError(full, retryable=True, **options)
    if status is not None and 400 <= status < 500:
        return InvalidRequestError(full, **options)
    return ProviderError(full, **options)

from __future__ import annotations

import importlib
import json
from typing import Any

from eurus.errors import ConfigError, ProviderError, ProviderUnavailableError
from eurus.providers.transport import HttpError, error_from_http

BODY_LIMIT = 2000


def import_sdk(module: str, extra: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ConfigError(
            f"the '{module}' package is not installed; run: pip install 'eurus-agent[{extra}]'"
        ) from exc


def _status_of(exc: Exception) -> int | None:
    for attribute in ("status_code", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool) and 100 <= value < 600:
            return value
    return None


def _header_map(response: Any) -> dict[str, str]:
    headers = getattr(response, "headers", None)
    if headers is None or not hasattr(headers, "items"):
        return {}
    return {str(key).lower(): str(value) for key, value in headers.items()}


def _retry_after(headers: dict[str, str]) -> float | None:
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _body_of(exc: Exception, response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        return text[:BODY_LIMIT]
    body = getattr(exc, "body", None)
    if isinstance(body, str) and body:
        return body[:BODY_LIMIT]
    if body is not None:
        return json.dumps(body, default=str)[:BODY_LIMIT]
    message = getattr(exc, "message", None)
    if isinstance(message, str) and message:
        return message[:BODY_LIMIT]
    return str(exc)[:BODY_LIMIT]


def error_from_sdk(provider: str, exc: Exception) -> ProviderError:
    status = _status_of(exc)
    if status is not None:
        response = getattr(exc, "response", None)
        headers = _header_map(response)
        error = HttpError(status, _body_of(exc, response), _retry_after(headers), headers)
        return error_from_http(provider, error)
    name = type(exc).__name__
    if "Timeout" in name or "Connection" in name:
        return ProviderUnavailableError(f"{provider}: {name}", provider=provider, retryable=True)
    return ProviderError(f"{provider}: {name}: {str(exc)[:200]}", provider=provider)


def dump_model(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    dumper = getattr(value, "model_dump", None)
    if callable(dumper):
        data = dumper()
        if isinstance(data, dict):
            return data
    raise ProviderError(f"unexpected SDK response type: {type(value).__name__}")

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

MASK = "REDACTED"
SENSITIVE_KEYS = (
    "key",
    "token",
    "password",
    "passwd",
    "secret",
    "cookie",
    "authorization",
    "credential",
)

_TOKENS = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|gsk_[A-Za-z0-9]{8,}|nvapi-[A-Za-z0-9_\-]{8,}"
    r"|AIza[0-9A-Za-z_\-]{20,}|ghp_[A-Za-z0-9]{20,}|(?i:bearer)\s+[A-Za-z0-9._\-]{8,})"
)
_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Za-z0-9_]*(?:api[_-]?key|token|secret|password|passwd)[A-Za-z0-9_]*)\s*([=:])\s*\S+"
)


def redact_text(text: str) -> str:
    text = _ASSIGNMENT.sub(lambda match: f"{match.group(1)}{match.group(2)}{MASK}", text)
    return _TOKENS.sub(MASK, text)


def redact_value(key: str, value: Any) -> Any:
    if any(token in key.lower() for token in SENSITIVE_KEYS):
        return MASK
    if isinstance(value, Mapping):
        return {str(name): redact_value(str(name), item) for name, item in value.items()}
    if isinstance(value, list | tuple):
        return [redact_value("", item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def redact_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {str(name): redact_value(str(name), item) for name, item in arguments.items()}


def summarize_arguments(arguments: Mapping[str, Any]) -> str:
    return json.dumps(redact_arguments(arguments), ensure_ascii=False, default=str)

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from eurus.catalog import ProviderSpec
from eurus.providers.httpx_transport import user_agent_candidates, with_user_agent

Fetch = Callable[[str, Mapping[str, str]], Awaitable[tuple[int, Any]]]

TIMEOUT_SECONDS = 10.0
ANTHROPIC_VERSION = "2023-06-01"
GEMINI_PROTOCOLS = ("gemini", "genai-sdk")
NON_CHAT_MARKERS = (
    "embed",
    "whisper",
    "tts",
    "dall-e",
    "moderation",
    "rerank",
    "transcribe",
    "guard",
    "image",
    "audio",
    "speech",
    "imagen",
    "veo",
)


@dataclass(frozen=True, slots=True)
class ModelInfo:
    id: str
    label: str = ""
    context: int = 0


@dataclass(frozen=True, slots=True)
class Discovery:
    provider: str
    models: tuple[ModelInfo, ...] = ()
    error: str = ""


def listing_request(spec: ProviderSpec, api_key: str | None) -> tuple[str, dict[str, str]]:
    headers = {"Accept": "application/json", **dict(spec.extra_headers)}
    if spec.user_agent:
        headers["User-Agent"] = spec.user_agent
    base = spec.base_url.rstrip("/")
    if spec.protocol == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
        if api_key:
            headers["x-api-key"] = api_key
        return f"{base}/v1/models?limit=1000", headers
    if spec.protocol in GEMINI_PROTOCOLS:
        if api_key:
            headers["x-goog-api-key"] = api_key
        return f"{base}/v1beta/models?pageSize=1000", headers
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return f"{base}/models", headers


def _is_chat_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return not any(marker in lowered for marker in NON_CHAT_MARKERS)


def _number(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def parse_models(protocol: str, payload: object) -> list[ModelInfo]:
    if not isinstance(payload, dict):
        return []
    found: dict[str, ModelInfo] = {}
    if protocol in GEMINI_PROTOCOLS:
        for row in payload.get("models") or []:
            if not isinstance(row, dict):
                continue
            methods = row.get("supportedGenerationMethods") or []
            if methods and "generateContent" not in methods:
                continue
            model_id = str(row.get("name", "")).removeprefix("models/")
            if model_id and _is_chat_model(model_id):
                label = str(row.get("displayName") or "")
                limit = _number(row.get("inputTokenLimit"))
                found.setdefault(model_id, ModelInfo(model_id, label, limit))
    else:
        for row in payload.get("data") or []:
            if not isinstance(row, dict):
                continue
            model_id = str(row.get("id") or "")
            if model_id and _is_chat_model(model_id):
                label = str(row.get("display_name") or row.get("name") or "")
                limit = _number(row.get("context_length"))
                found.setdefault(model_id, ModelInfo(model_id, label, limit))
    return sorted(found.values(), key=lambda info: info.id.lower())


def describe_status(status: int) -> str:
    if status in (401, 403):
        return f"key rejected or not allowed (HTTP {status})"
    if status == 404:
        return "this provider has no model listing endpoint (HTTP 404)"
    if status == 429:
        return "rate limited (HTTP 429)"
    if status >= 500:
        return f"provider error (HTTP {status})"
    return f"unexpected response (HTTP {status})"


async def default_fetch(url: str, headers: Mapping[str, str]) -> tuple[int, Any]:
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=False) as client:
        response = await client.get(url, headers=dict(headers))
    try:
        payload = response.json()
    except ValueError:
        payload = None
    return response.status_code, payload


async def discover(
    spec: ProviderSpec, api_key: str | None, fetch: Fetch | None = None
) -> Discovery:
    if spec.api_key_env and not api_key:
        return Discovery(spec.name, error="no API key")
    url, headers = listing_request(spec, api_key)
    request = fetch or default_fetch
    status = 0
    payload: Any = None
    try:
        for agent in user_agent_candidates(headers):
            status, payload = await request(url, with_user_agent(headers, agent))
            if status != 403:
                break
    except (httpx.HTTPError, ValueError) as exc:
        return Discovery(spec.name, error=f"network error: {type(exc).__name__}")
    if status >= 400 or status == 0:
        return Discovery(spec.name, error=describe_status(status))
    models = parse_models(spec.protocol, payload)
    if not models:
        return Discovery(spec.name, error="the provider returned no chat models")
    return Discovery(spec.name, tuple(models))

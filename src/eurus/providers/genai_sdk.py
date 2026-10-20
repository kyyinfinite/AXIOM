from __future__ import annotations

import base64
import contextlib
import inspect
from collections.abc import AsyncIterator, Sequence
from typing import Any

from eurus.catalog import ProviderSpec
from eurus.errors import ProviderError
from eurus.providers.base import (
    ChatRequest,
    ChatResponse,
    Provider,
    RetryPolicy,
    Sleep,
    StreamEvent,
)
from eurus.providers.gemini import (
    SIGNATURE_KEY,
    StreamAssembler,
    convert_messages,
    parse_response,
    tool_to_wire,
)
from eurus.providers.sdk import error_from_sdk, import_sdk
from eurus.providers.transport import HttpTransport

DEFAULT_BASE = "https://generativelanguage.googleapis.com"


def _signature_text(value: Any) -> str:
    if isinstance(value, bytes | bytearray):
        return base64.b64encode(bytes(value)).decode("ascii")
    return value if isinstance(value, str) else ""


def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", value))


def _part_to_rest(part: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    text = getattr(part, "text", None)
    if isinstance(text, str):
        out["text"] = text
    if getattr(part, "thought", None) is True:
        out["thought"] = True
    call = getattr(part, "function_call", None)
    if call is not None:
        args = getattr(call, "args", None)
        out["functionCall"] = {
            "name": str(getattr(call, "name", "") or ""),
            "args": dict(args) if isinstance(args, dict) else {},
        }
    signature = _signature_text(getattr(part, "thought_signature", None))
    if signature:
        out[SIGNATURE_KEY] = signature
    return out


def response_to_rest(response: Any) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        parts = [_part_to_rest(part) for part in getattr(content, "parts", None) or []]
        entry: dict[str, Any] = {"content": {"parts": parts}}
        reason = getattr(candidate, "finish_reason", None)
        if reason is not None:
            entry["finishReason"] = _enum_name(reason)
        candidates.append(entry)
    data: dict[str, Any] = {"candidates": candidates}
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        data["usageMetadata"] = {
            "promptTokenCount": getattr(usage, "prompt_token_count", 0),
            "candidatesTokenCount": getattr(usage, "candidates_token_count", 0),
            "thoughtsTokenCount": getattr(usage, "thoughts_token_count", 0),
        }
    block = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
    if block is not None:
        data["promptFeedback"] = {"blockReason": _enum_name(block)}
    version = getattr(response, "model_version", None)
    if isinstance(version, str) and version:
        data["modelVersion"] = version
    return data


def decode_signatures(contents: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for content in contents:
        parts: list[dict[str, Any]] = []
        for part in content.get("parts") or []:
            item = dict(part)
            raw = item.pop(SIGNATURE_KEY, None)
            if isinstance(raw, str) and raw:
                with contextlib.suppress(ValueError):
                    item["thoughtSignature"] = base64.b64decode(raw, validate=True)
            parts.append(item)
        prepared.append({**content, "parts": parts})
    return prepared


def build_config(request: ChatRequest, system: str) -> dict[str, Any]:
    config: dict[str, Any] = {"automatic_function_calling": {"disable": True}}
    if system:
        config["system_instruction"] = system
    if request.temperature is not None:
        config["temperature"] = request.temperature
    if request.max_tokens is not None:
        config["max_output_tokens"] = request.max_tokens
    if request.tools:
        config["tools"] = [{"function_declarations": [tool_to_wire(t) for t in request.tools]}]
    return config


class GenaiSdkProvider(Provider):
    def __init__(
        self,
        spec: ProviderSpec,
        api_key: str | None,
        transport: HttpTransport,
        retry: RetryPolicy | None = None,
        sleep: Sleep | None = None,
        *,
        timeout: float = 120.0,
        client: Any = None,
    ) -> None:
        super().__init__(spec, api_key, transport, retry, sleep)
        self._timeout = timeout
        self._client: Any = client
        self._sdk: Any = None if client is not None else import_sdk("google.genai", "genai")

    def _get_client(self) -> Any:
        if self._client is None:
            options: dict[str, Any] = {"timeout": int(self._timeout * 1000)}
            if self.spec.base_url.rstrip("/") != DEFAULT_BASE:
                options["base_url"] = self.spec.base_url.rstrip("/")
            headers = self._extra_headers()
            if headers:
                options["headers"] = headers
            self._client = self._sdk.Client(api_key=self._key(), http_options=options)
        return self._client

    def _arguments(self, request: ChatRequest) -> dict[str, Any]:
        system, contents = convert_messages(request.messages)
        return {
            "model": request.model,
            "contents": decode_signatures(contents),
            "config": build_config(request, system),
        }

    async def _chat(self, request: ChatRequest) -> ChatResponse:
        client = self._get_client()
        arguments = self._arguments(request)
        try:
            response = await client.aio.models.generate_content(**arguments)
            data = response_to_rest(response)
        except ProviderError:
            raise
        except Exception as exc:
            raise error_from_sdk(self.name, exc) from exc
        return parse_response(data, request.model, self.name)

    async def _stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        client = self._get_client()
        arguments = self._arguments(request)
        assembler = StreamAssembler(self.name)
        try:
            stream = client.aio.models.generate_content_stream(**arguments)
            if inspect.isawaitable(stream):
                stream = await stream
            async for chunk in stream:
                for event in assembler.feed(response_to_rest(chunk)):
                    yield event
        except ProviderError:
            raise
        except Exception as exc:
            raise error_from_sdk(self.name, exc) from exc
        for event in assembler.close():
            yield event

    async def aclose(self) -> None:
        aio = getattr(self._client, "aio", None)
        closer = getattr(aio, "aclose", None)
        if callable(closer):
            await closer()

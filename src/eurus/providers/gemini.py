from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any

from eurus.errors import ProviderUnavailableError
from eurus.models import Message, ToolCall, Usage
from eurus.providers.base import (
    ChatRequest,
    ChatResponse,
    Provider,
    StreamEvent,
    ToolSpec,
    as_int,
    parse_arguments,
)
from eurus.providers.transport import error_from_payload

SCHEMA_KEYS = frozenset(
    {
        "type",
        "description",
        "properties",
        "required",
        "items",
        "enum",
        "format",
        "nullable",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "anyOf",
    }
)
FINISH_REASONS = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
}
SIGNATURE_KEY = "thoughtSignature"


def clean_schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in SCHEMA_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {name: clean_schema(sub) for name, sub in value.items()}
        elif key == "items":
            cleaned[key] = clean_schema(value)
        elif key == "anyOf" and isinstance(value, list):
            cleaned[key] = [clean_schema(item) for item in value]
        else:
            cleaned[key] = value
    return cleaned


def tool_to_wire(spec: ToolSpec) -> dict[str, Any]:
    declaration: dict[str, Any] = {"name": spec.name, "description": spec.description}
    parameters = clean_schema(spec.parameters)
    if isinstance(parameters, dict) and parameters.get("properties"):
        declaration["parameters"] = parameters
    return declaration


def convert_messages(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    for message in messages:
        if message.role == "system":
            if message.content:
                system_parts.append(message.content)
        elif message.role == "user":
            contents.append({"role": "user", "parts": [{"text": message.content}]})
        elif message.role == "assistant":
            parts: list[dict[str, Any]] = []
            if message.content:
                parts.append({"text": message.content})
            for call in message.tool_calls:
                names[call.id] = call.name
                part: dict[str, Any] = {
                    "functionCall": {"name": call.name, "args": dict(call.arguments)}
                }
                if SIGNATURE_KEY in call.extra:
                    part[SIGNATURE_KEY] = call.extra[SIGNATURE_KEY]
                parts.append(part)
            if parts:
                contents.append({"role": "model", "parts": parts})
        else:
            name = message.name or names.get(message.tool_call_id or "", "tool")
            response_part = {
                "functionResponse": {"name": name, "response": {"result": message.content}}
            }
            last = contents[-1] if contents else None
            if (
                last is not None
                and last["role"] == "user"
                and all("functionResponse" in item for item in last["parts"])
            ):
                last["parts"].append(response_part)
            else:
                contents.append({"role": "user", "parts": [response_part]})
    return "\n\n".join(system_parts), contents


def _read_parts(parts: Sequence[dict[str, Any]]) -> tuple[str, list[ToolCall]]:
    texts: list[str] = []
    calls: list[ToolCall] = []
    for part in parts:
        call = part.get("functionCall")
        if isinstance(call, dict):
            extra = {SIGNATURE_KEY: part[SIGNATURE_KEY]} if part.get(SIGNATURE_KEY) else {}
            calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    name=str(call.get("name") or ""),
                    arguments=parse_arguments(call.get("args")),
                    extra=extra,
                )
            )
        elif isinstance(part.get("text"), str) and not part.get("thought"):
            texts.append(part["text"])
    return "".join(texts), calls


def _usage(raw: Any) -> Usage | None:
    if not isinstance(raw, dict):
        return None
    output = as_int(raw.get("candidatesTokenCount")) + as_int(raw.get("thoughtsTokenCount"))
    return Usage(as_int(raw.get("promptTokenCount")), output)


def _finish(reason: Any, has_calls: bool) -> str:
    if has_calls:
        return "tool_calls"
    if not reason:
        return "stop"
    return FINISH_REASONS.get(str(reason), "other")


def build_payload(request: ChatRequest) -> dict[str, Any]:
    system, contents = convert_messages(request.messages)
    payload: dict[str, Any] = {"contents": contents}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    config: dict[str, Any] = {}
    if request.temperature is not None:
        config["temperature"] = request.temperature
    if request.max_tokens is not None:
        config["maxOutputTokens"] = request.max_tokens
    if config:
        payload["generationConfig"] = config
    if request.tools:
        payload["tools"] = [{"functionDeclarations": [tool_to_wire(t) for t in request.tools]}]
    return payload


def parse_response(data: dict[str, Any], model: str, provider: str) -> ChatResponse:
    usage = _usage(data.get("usageMetadata")) or Usage()
    name = str(data.get("modelVersion") or model)
    candidates = data.get("candidates") or []
    if not candidates:
        error = error_from_payload(provider, data)
        if error is not None:
            raise error
        blocked = (data.get("promptFeedback") or {}).get("blockReason")
        return ChatResponse(
            message=Message.assistant(""),
            usage=usage,
            finish_reason="content_filter" if blocked else "other",
            model=name,
            provider=provider,
        )
    candidate = candidates[0]
    text, calls = _read_parts((candidate.get("content") or {}).get("parts") or [])
    return ChatResponse(
        message=Message.assistant(text, tuple(calls)),
        usage=usage,
        finish_reason=_finish(candidate.get("finishReason"), bool(calls)),
        model=name,
        provider=provider,
    )


class StreamAssembler:
    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.usage: Usage | None = None
        self.reason: Any = None
        self.saw_tool = False
        self.chunks = 0

    def feed(self, chunk: dict[str, Any]) -> list[StreamEvent]:
        self.chunks += 1
        if not chunk.get("candidates"):
            error = error_from_payload(self.provider, chunk)
            if error is not None:
                raise error
        events: list[StreamEvent] = []
        self.usage = _usage(chunk.get("usageMetadata")) or self.usage
        for candidate in chunk.get("candidates") or []:
            text, calls = _read_parts((candidate.get("content") or {}).get("parts") or [])
            if text:
                events.append(StreamEvent.text_delta(text))
            for call in calls:
                self.saw_tool = True
                events.append(StreamEvent.tool(call))
            self.reason = candidate.get("finishReason") or self.reason
        return events

    def close(self) -> list[StreamEvent]:
        if self.chunks == 0:
            raise ProviderUnavailableError(
                f"{self.provider}: stream ended without any data",
                provider=self.provider,
                retryable=True,
            )
        events: list[StreamEvent] = []
        if self.usage is not None:
            events.append(StreamEvent.usage_report(self.usage))
        events.append(StreamEvent.done(_finish(self.reason, self.saw_tool)))
        return events


class GeminiProvider(Provider):
    def _url(self, model: str, stream: bool) -> str:
        name = model.removeprefix("models/")
        base = f"{self.spec.base_url}/v1beta/models/{name}"
        if stream:
            return f"{base}:streamGenerateContent?alt=sse"
        return f"{base}:generateContent"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        headers.update(self._extra_headers())
        key = self._key()
        if key:
            headers["x-goog-api-key"] = key
        return headers

    def _payload(self, request: ChatRequest) -> dict[str, Any]:
        return build_payload(request)

    async def _chat(self, request: ChatRequest) -> ChatResponse:
        data = await self._post(
            self._url(request.model, False), self._headers(), self._payload(request)
        )
        return parse_response(data, request.model, self.name)

    async def _stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        url = self._url(request.model, True)
        assembler = StreamAssembler(self.name)
        async for data in self._sse(url, self._headers(), self._payload(request)):
            for event in assembler.feed(self._loads(data)):
                yield event
        for event in assembler.close():
            yield event

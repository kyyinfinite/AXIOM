from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any
from urllib.parse import urlparse

from eurus.catalog import ProviderSpec
from eurus.errors import InvalidRequestError, ProviderError, ProviderUnavailableError
from eurus.models import Message, ToolCall, Usage
from eurus.providers.base import (
    ChatRequest,
    ChatResponse,
    Provider,
    RetryPolicy,
    Sleep,
    StreamEvent,
    ToolSpec,
    as_int,
    parse_arguments,
)
from eurus.providers.streams import closing_stream
from eurus.providers.transport import HttpTransport, error_from_payload

STREAM_USAGE_PROVIDERS = frozenset({"openai", "groq", "deepseek"})
STREAM_USAGE_HOSTS = frozenset({"api.openai.com", "api.groq.com", "api.deepseek.com"})
COALESCE_SYSTEM_HOSTS = frozenset({"integrate.api.nvidia.com"})
MAX_ADAPTATIONS = 2
FINISH_REASONS = {
    "stop": "stop",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "length": "length",
    "content_filter": "content_filter",
}


def message_to_wire(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id or "",
            "content": message.content,
        }
    wire: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        wire["content"] = message.content or None
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
    return wire


def tool_to_wire(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def coalesce_system(messages: Sequence[Message]) -> list[Message]:
    system = [m.content for m in messages if m.role == "system" and m.content]
    rest = [message for message in messages if message.role != "system"]
    if not system:
        return rest
    return [Message.system("\n\n".join(system)), *rest]


def _usage(raw: Any) -> Usage:
    if not isinstance(raw, dict):
        return Usage()
    return Usage(as_int(raw.get("prompt_tokens")), as_int(raw.get("completion_tokens")))


def _finish(reason: Any, has_calls: bool) -> str:
    if has_calls:
        return "tool_calls"
    if not reason:
        return "stop"
    return FINISH_REASONS.get(str(reason), "other")


def _tool_call(item: dict[str, Any], index: int) -> ToolCall:
    function = item.get("function") or {}
    return ToolCall(
        id=str(item.get("id") or f"call_{index}"),
        name=str(function.get("name") or ""),
        arguments=parse_arguments(function.get("arguments")),
    )


def parse_chat(data: dict[str, Any], provider: str, model: str) -> ChatResponse:
    choices = data.get("choices") or []
    if not choices:
        error = error_from_payload(provider, data)
        if error is not None:
            raise error
        raise ProviderError(f"{provider}: response has no choices", provider=provider)
    choice = choices[0]
    raw = choice.get("message") or {}
    content = raw.get("content")
    calls = tuple(_tool_call(item, index) for index, item in enumerate(raw.get("tool_calls") or []))
    return ChatResponse(
        message=Message.assistant(content if isinstance(content, str) else "", calls),
        usage=_usage(data.get("usage")),
        finish_reason=_finish(choice.get("finish_reason"), bool(calls)),
        model=str(data.get("model") or model),
        provider=provider,
    )


class StreamAssembler:
    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.pending: dict[int, dict[str, str]] = {}
        self.usage: Usage | None = None
        self.reason: Any = None
        self.chunks = 0

    def feed(self, chunk: dict[str, Any]) -> list[StreamEvent]:
        self.chunks += 1
        if not chunk.get("choices"):
            error = error_from_payload(self.provider, chunk)
            if error is not None:
                raise error
        events: list[StreamEvent] = []
        raw_usage = chunk.get("usage")
        if not isinstance(raw_usage, dict):
            extension = chunk.get("x_groq")
            raw_usage = extension.get("usage") if isinstance(extension, dict) else None
        if isinstance(raw_usage, dict):
            self.usage = _usage(raw_usage)
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if isinstance(text, str) and text:
                events.append(StreamEvent.text_delta(text))
            for item in delta.get("tool_calls") or []:
                self._merge_call(item)
            if choice.get("finish_reason"):
                self.reason = choice["finish_reason"]
        return events

    def _merge_call(self, item: dict[str, Any]) -> None:
        index = item.get("index")
        if not isinstance(index, int):
            index = len(self.pending)
        slot = self.pending.setdefault(index, {"id": "", "name": "", "arguments": ""})
        if item.get("id"):
            slot["id"] = str(item["id"])
        function = item.get("function") or {}
        if function.get("name") and not slot["name"]:
            slot["name"] = str(function["name"])
        slot["arguments"] += str(function.get("arguments") or "")

    def close(self) -> list[StreamEvent]:
        if self.chunks == 0:
            raise ProviderUnavailableError(
                f"{self.provider}: stream ended without any data",
                provider=self.provider,
                retryable=True,
            )
        events: list[StreamEvent] = []
        for index in sorted(self.pending):
            slot = self.pending[index]
            if not slot["name"]:
                continue
            call = ToolCall(
                id=slot["id"] or f"call_{index}",
                name=slot["name"],
                arguments=parse_arguments(slot["arguments"]),
            )
            events.append(StreamEvent.tool(call))
        has_calls = bool(events)
        if self.usage is not None:
            events.append(StreamEvent.usage_report(self.usage))
        events.append(StreamEvent.done(_finish(self.reason, has_calls)))
        return events


class OpenAICompatProvider(Provider):
    def __init__(
        self,
        spec: ProviderSpec,
        api_key: str | None,
        transport: HttpTransport,
        retry: RetryPolicy | None = None,
        sleep: Sleep | None = None,
    ) -> None:
        super().__init__(spec, api_key, transport, retry, sleep)
        self._token_param = "max_tokens"
        self._send_temperature = True

    @property
    def _host(self) -> str:
        return (urlparse(self.spec.base_url).hostname or "").lower()

    def _url(self) -> str:
        return f"{self.spec.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        headers.update(self._extra_headers())
        key = self._key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _wants_stream_usage(self) -> bool:
        return self.name in STREAM_USAGE_PROVIDERS or self._host in STREAM_USAGE_HOSTS

    def _payload(self, request: ChatRequest, stream: bool) -> dict[str, Any]:
        messages = request.messages
        if self._host in COALESCE_SYSTEM_HOSTS:
            messages = tuple(coalesce_system(messages))
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [message_to_wire(message) for message in messages],
        }
        if request.temperature is not None and self._send_temperature:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload[self._token_param] = request.max_tokens
        if request.tools:
            payload["tools"] = [tool_to_wire(tool) for tool in request.tools]
        if stream:
            payload["stream"] = True
            if self._wants_stream_usage():
                payload["stream_options"] = {"include_usage": True}
        return payload

    def _adapt(self, error: ProviderError) -> bool:
        text = str(error).lower()
        if (
            self._token_param == "max_tokens"
            and "max_tokens" in text
            and "max_completion_tokens" in text
        ):
            self._token_param = "max_completion_tokens"
            return True
        rejects = ("unsupported", "not support", "only the default", "does not support")
        if self._send_temperature and "temperature" in text and any(w in text for w in rejects):
            self._send_temperature = False
            return True
        return False

    async def _call(self, request: ChatRequest) -> dict[str, Any]:
        for _ in range(MAX_ADAPTATIONS):
            try:
                return await self._post(self._url(), self._headers(), self._payload(request, False))
            except InvalidRequestError as exc:
                if not self._adapt(exc):
                    raise
        return await self._post(self._url(), self._headers(), self._payload(request, False))

    async def _chat(self, request: ChatRequest) -> ChatResponse:
        data = await self._call(request)
        return parse_chat(data, self.name, request.model)

    async def _stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        assembler = StreamAssembler(self.name)
        for attempt in range(MAX_ADAPTATIONS + 1):
            payload = self._payload(request, True)
            started = False
            try:
                stream = self._sse(self._url(), self._headers(), payload)
                async with closing_stream(stream) as chunks:
                    async for data in chunks:
                        if data.strip() == "[DONE]":
                            break
                        started = True
                        for event in assembler.feed(self._loads(data)):
                            yield event
            except InvalidRequestError as exc:
                if started or attempt >= MAX_ADAPTATIONS or not self._adapt(exc):
                    raise
                continue
            break
        for event in assembler.close():
            yield event

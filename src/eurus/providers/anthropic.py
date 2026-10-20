from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator, Sequence
from typing import Any

from eurus.errors import (
    AuthenticationError,
    InvalidRequestError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
)
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

log = logging.getLogger(__name__)

API_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096
STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "refusal": "content_filter",
}
STREAM_ERRORS: dict[str, tuple[type[ProviderError], bool]] = {
    "overloaded_error": (ProviderUnavailableError, True),
    "api_error": (ProviderUnavailableError, True),
    "rate_limit_error": (RateLimitError, True),
    "authentication_error": (AuthenticationError, False),
    "permission_error": (AuthenticationError, False),
    "invalid_request_error": (InvalidRequestError, False),
}


def convert_messages(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    wire: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            if message.content:
                system_parts.append(message.content)
        elif message.role == "user":
            wire.append({"role": "user", "content": message.content})
        elif message.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            for call in message.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": dict(call.arguments),
                    }
                )
            if blocks:
                wire.append({"role": "assistant", "content": blocks})
        else:
            block = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id or "",
                "content": message.content,
            }
            last = wire[-1] if wire else None
            if (
                last is not None
                and last["role"] == "user"
                and isinstance(last["content"], list)
                and all(item.get("type") == "tool_result" for item in last["content"])
            ):
                last["content"].append(block)
            else:
                wire.append({"role": "user", "content": [block]})
    return "\n\n".join(system_parts), wire


def tool_to_wire(spec: ToolSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.parameters,
    }


class AnthropicProvider(Provider):
    def _url(self) -> str:
        return f"{self.spec.base_url}/v1/messages"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "anthropic-version": API_VERSION}
        headers.update(self._extra_headers())
        key = self._key()
        if key:
            headers["x-api-key"] = key
        return headers

    def _payload(self, request: ChatRequest, stream: bool) -> dict[str, Any]:
        system, wire = convert_messages(request.messages)
        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens or DEFAULT_MAX_TOKENS,
            "messages": wire,
        }
        if system:
            system_blocks: list[dict[str, Any]] = []
            pattern = r"<eurus-tier:(stable|context|volatile)>\n(.*?)\n</eurus-tier:\1>"
            matches = list(re.finditer(pattern, system, re.DOTALL))
            if matches:
                for match in matches:
                    block: dict[str, Any] = {"type": "text", "text": match.group(2)}
                    if match.group(1) == "stable":
                        block["cache_control"] = {"type": "ephemeral"}
                    system_blocks.append(block)
                remainder = re.sub(pattern, "", system, flags=re.DOTALL).strip()
                if remainder:
                    system_blocks.append({"type": "text", "text": remainder})
            else:
                payload["system"] = system
            if matches:
                payload["system"] = system_blocks
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.tools:
            payload["tools"] = [tool_to_wire(tool) for tool in request.tools]
        if stream:
            payload["stream"] = True
        return payload

    async def _chat(self, request: ChatRequest) -> ChatResponse:
        data = await self._post(self._url(), self._headers(), self._payload(request, False))
        blocks = data.get("content") or []
        text = "".join(str(b.get("text", "")) for b in blocks if b.get("type") == "text")
        calls = tuple(
            ToolCall(
                id=str(b.get("id", "")),
                name=str(b.get("name", "")),
                arguments=parse_arguments(b.get("input")),
            )
            for b in blocks
            if b.get("type") == "tool_use"
        )
        usage = data.get("usage") or {}
        cache_creation = as_int(usage.get("cache_creation_input_tokens"))
        cache_read = as_int(usage.get("cache_read_input_tokens"))
        if cache_creation or cache_read:
            log.debug(
                "Anthropic prompt cache creation_tokens=%d read_tokens=%d",
                cache_creation,
                cache_read,
            )
        stop = data.get("stop_reason")
        if calls:
            finish = "tool_calls"
        elif stop:
            finish = STOP_REASONS.get(str(stop), "other")
        else:
            finish = "stop"
        return ChatResponse(
            message=Message.assistant(text, calls),
            usage=Usage(
                as_int(usage.get("input_tokens")),
                as_int(usage.get("output_tokens")),
                as_int(usage.get("cache_creation_input_tokens")),
                as_int(usage.get("cache_read_input_tokens")),
            ),
            finish_reason=finish,
            model=str(data.get("model") or request.model),
            provider=self.name,
        )

    def _stream_error(self, event: dict[str, Any]) -> ProviderError:
        error = event.get("error") or {}
        kind, retryable = STREAM_ERRORS.get(str(error.get("type")), (ProviderError, False))
        message = f"{self.name}: {error.get('message') or 'stream error'}"
        return kind(message, provider=self.name, retryable=retryable)

    async def _stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        payload = self._payload(request, True)
        tools: dict[int, dict[str, str]] = {}
        input_tokens = 0
        output_tokens = 0
        cache_creation_tokens = 0
        cache_read_tokens = 0
        stop: Any = None
        saw_tool = False
        async for data in self._sse(self._url(), self._headers(), payload):
            event = self._loads(data)
            kind = event.get("type")
            index = as_int(event.get("index"))
            if kind == "message_start":
                usage = (event.get("message") or {}).get("usage") or {}
                input_tokens = as_int(usage.get("input_tokens"))
                output_tokens = as_int(usage.get("output_tokens"))
                cache_creation_tokens = as_int(usage.get("cache_creation_input_tokens"))
                cache_read_tokens = as_int(usage.get("cache_read_input_tokens"))
            elif kind == "content_block_start":
                block = event.get("content_block") or {}
                if block.get("type") == "tool_use":
                    tools[index] = {
                        "id": str(block.get("id", "")),
                        "name": str(block.get("name", "")),
                        "json": "",
                    }
            elif kind == "content_block_delta":
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta":
                    text = delta.get("text")
                    if isinstance(text, str) and text:
                        yield StreamEvent.text_delta(text)
                elif delta.get("type") == "input_json_delta" and index in tools:
                    tools[index]["json"] += str(delta.get("partial_json") or "")
            elif kind == "content_block_stop":
                slot = tools.pop(index, None)
                if slot is not None:
                    saw_tool = True
                    yield StreamEvent.tool(
                        ToolCall(slot["id"], slot["name"], parse_arguments(slot["json"]))
                    )
            elif kind == "message_delta":
                delta = event.get("delta") or {}
                stop = delta.get("stop_reason") or stop
                usage = event.get("usage") or {}
                output_tokens = as_int(usage.get("output_tokens")) or output_tokens
                cache_creation_tokens = (
                    as_int(usage.get("cache_creation_input_tokens")) or cache_creation_tokens
                )
                cache_read_tokens = (
                    as_int(usage.get("cache_read_input_tokens")) or cache_read_tokens
                )
            elif kind == "error":
                raise self._stream_error(event)
            elif kind == "message_stop":
                break
        if cache_creation_tokens or cache_read_tokens:
            log.debug(
                "Anthropic prompt cache creation_tokens=%d read_tokens=%d",
                cache_creation_tokens,
                cache_read_tokens,
            )
        yield StreamEvent.usage_report(
            Usage(input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens)
        )
        if saw_tool:
            finish = "tool_calls"
        elif stop:
            finish = STOP_REASONS.get(str(stop), "other")
        else:
            finish = "stop"
        yield StreamEvent.done(finish)

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from axiom.catalog import ProviderSpec
from axiom.errors import AuthenticationError, ProviderError, ProviderUnavailableError
from axiom.models import Message, ToolCall, Usage
from axiom.providers.transport import (
    HttpError,
    HttpTransport,
    TransportError,
    error_from_http,
)

INVALID_ARGUMENTS_KEY = "_invalid_arguments"

Sleep = Callable[[float], Awaitable[None]]
EventKind = Literal["text", "tool_call", "usage", "done"]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})


@dataclass(frozen=True, slots=True)
class ChatRequest:
    model: str
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ChatResponse:
    message: Message
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"
    model: str = ""
    provider: str = ""


@dataclass(frozen=True, slots=True)
class StreamEvent:
    kind: EventKind
    text: str = ""
    tool_call: ToolCall | None = None
    usage: Usage | None = None
    finish_reason: str = ""

    @classmethod
    def text_delta(cls, text: str) -> StreamEvent:
        return cls("text", text=text)

    @classmethod
    def tool(cls, call: ToolCall) -> StreamEvent:
        return cls("tool_call", tool_call=call)

    @classmethod
    def usage_report(cls, usage: Usage) -> StreamEvent:
        return cls("usage", usage=usage)

    @classmethod
    def done(cls, finish_reason: str) -> StreamEvent:
        return cls("done", finish_reason=finish_reason)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    max_retry_after: float = 30.0

    def delay(self, attempt: int, retry_after: float | None) -> float:
        value = min(self.max_delay, self.base_delay * (2**attempt))
        if retry_after is not None:
            value = max(value, min(retry_after, self.max_retry_after))
        return float(value)


def as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if raw is None:
        return {}
    text = str(raw).strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except ValueError:
        return {INVALID_ARGUMENTS_KEY: text}
    if isinstance(value, dict):
        return value
    return {INVALID_ARGUMENTS_KEY: text}


async def collect(
    events: AsyncIterator[StreamEvent], *, provider: str = "", model: str = ""
) -> ChatResponse:
    parts: list[str] = []
    calls: list[ToolCall] = []
    usage = Usage()
    finish = "stop"
    async for event in events:
        if event.kind == "text":
            parts.append(event.text)
        elif event.kind == "tool_call" and event.tool_call is not None:
            calls.append(event.tool_call)
        elif event.kind == "usage" and event.usage is not None:
            usage = event.usage
        elif event.kind == "done":
            finish = event.finish_reason or finish
    return ChatResponse(
        message=Message.assistant("".join(parts), tuple(calls)),
        usage=usage,
        finish_reason=finish,
        model=model,
        provider=provider,
    )


class Provider(ABC):
    def __init__(
        self,
        spec: ProviderSpec,
        api_key: str | None,
        transport: HttpTransport,
        retry: RetryPolicy | None = None,
        sleep: Sleep | None = None,
    ) -> None:
        self.spec = spec
        self._api_key = api_key
        self.transport = transport
        self.retry = retry or RetryPolicy()
        self._sleep: Sleep = sleep or asyncio.sleep

    @property
    def name(self) -> str:
        return self.spec.name

    def _extra_headers(self) -> dict[str, str]:
        headers = dict(self.spec.extra_headers)
        if self.spec.user_agent:
            headers["User-Agent"] = self.spec.user_agent
        return headers

    async def aclose(self) -> None:
        return None

    def _key(self) -> str | None:
        if self.spec.api_key_env and not self._api_key:
            raise AuthenticationError(
                f"{self.name}: environment variable {self.spec.api_key_env} is not set",
                provider=self.name,
            )
        return self._api_key

    async def chat(self, request: ChatRequest) -> ChatResponse:
        attempt = 0
        while True:
            try:
                return await self._chat(request)
            except ProviderError as exc:
                if not exc.retryable or attempt >= self.retry.max_retries:
                    raise
                await self._sleep(self.retry.delay(attempt, exc.retry_after))
                attempt += 1

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        attempt = 0
        while True:
            started = False
            try:
                async for event in self._stream(request):
                    started = True
                    yield event
                return
            except ProviderError as exc:
                if started or not exc.retryable or attempt >= self.retry.max_retries:
                    raise
                await self._sleep(self.retry.delay(attempt, exc.retry_after))
                attempt += 1

    @abstractmethod
    async def _chat(self, request: ChatRequest) -> ChatResponse: ...

    @abstractmethod
    def _stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]: ...

    async def _post(
        self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        try:
            return await self.transport.post_json(url, headers, payload)
        except HttpError as exc:
            raise error_from_http(self.name, exc) from exc
        except TransportError as exc:
            raise ProviderUnavailableError(
                f"{self.name}: {exc}", provider=self.name, retryable=True
            ) from exc

    async def _sse(
        self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any]
    ) -> AsyncIterator[str]:
        try:
            async for data in self.transport.stream_sse(url, headers, payload):
                yield data
        except HttpError as exc:
            raise error_from_http(self.name, exc) from exc
        except TransportError as exc:
            raise ProviderUnavailableError(
                f"{self.name}: {exc}", provider=self.name, retryable=True
            ) from exc

    def _loads(self, data: str) -> dict[str, Any]:
        try:
            value = json.loads(data)
        except ValueError as exc:
            raise ProviderError(f"{self.name}: malformed stream chunk", provider=self.name) from exc
        if not isinstance(value, dict):
            raise ProviderError(f"{self.name}: unexpected stream chunk", provider=self.name)
        return value

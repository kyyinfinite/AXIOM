from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence

from eurus.errors import AllProvidersFailedError, EurusError, ProviderError
from eurus.models import Message, ModelRef
from eurus.providers.base import ChatRequest, ChatResponse, Provider, StreamEvent, ToolSpec
from eurus.providers.streams import closing_stream
from eurus.providers.transport import HttpTransport


class Router:
    def __init__(
        self,
        providers: Mapping[str, Provider],
        chain: Sequence[ModelRef],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        transport: HttpTransport | None = None,
    ) -> None:
        if not chain:
            raise ValueError("router needs at least one model")
        self.providers = dict(providers)
        self.chain = tuple(chain)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_used: ModelRef | None = None
        self._transport = transport

    def _request(
        self,
        ref: ModelRef,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        temperature: float | None,
        max_tokens: int | None,
    ) -> ChatRequest:
        return ChatRequest(
            model=ref.model,
            messages=tuple(messages),
            tools=tuple(tools),
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )

    @staticmethod
    def _failure(failures: Sequence[tuple[str, ProviderError]]) -> EurusError:
        if len(failures) == 1:
            return failures[0][1]
        return AllProvidersFailedError(failures)

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        failures: list[tuple[str, ProviderError]] = []
        for ref in self.chain:
            request = self._request(ref, messages, tools, temperature, max_tokens)
            try:
                response = await self.providers[ref.provider].chat(request)
            except ProviderError as exc:
                failures.append((str(ref), exc))
                continue
            self.last_used = ref
            return response
        raise self._failure(failures)

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        failures: list[tuple[str, ProviderError]] = []
        for ref in self.chain:
            request = self._request(ref, messages, tools, temperature, max_tokens)
            started = False
            try:
                async with closing_stream(self.providers[ref.provider].stream(request)) as events:
                    async for event in events:
                        if not started:
                            started = True
                            self.last_used = ref
                        yield event
                self.last_used = ref
                return
            except ProviderError as exc:
                if started:
                    raise
                failures.append((str(ref), exc))
        raise self._failure(failures)

    async def aclose(self) -> None:
        for provider in self.providers.values():
            await provider.aclose()
        if self._transport is not None:
            await self._transport.aclose()

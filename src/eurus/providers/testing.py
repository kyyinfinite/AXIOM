from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from eurus.catalog import BUILTIN_PROVIDERS
from eurus.models import ModelRef
from eurus.providers.base import RetryPolicy
from eurus.providers.openai_compat import OpenAICompatProvider
from eurus.providers.router import Router


@dataclass(frozen=True, slots=True)
class RecordedCall:
    url: str
    headers: dict[str, str]
    payload: dict[str, Any]


class FakeTransport:
    def __init__(
        self,
        json_responses: Sequence[dict[str, Any] | Exception] = (),
        streams: Sequence[Sequence[str | Exception] | Exception] = (),
    ) -> None:
        self._json = list(json_responses)
        self._streams = list(streams)
        self.calls: list[RecordedCall] = []
        self.closed = False

    async def post_json(
        self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        self.calls.append(RecordedCall(url, dict(headers), dict(payload)))
        if not self._json:
            raise AssertionError("no scripted json response left")
        item = self._json.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def stream_sse(
        self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any]
    ) -> AsyncIterator[str]:
        self.calls.append(RecordedCall(url, dict(headers), dict(payload)))
        if not self._streams:
            raise AssertionError("no scripted stream left")
        item = self._streams.pop(0)
        if isinstance(item, Exception):
            raise item
        for entry in item:
            if isinstance(entry, Exception):
                raise entry
            yield entry

    async def aclose(self) -> None:
        self.closed = True


class RecordingSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


ScriptedCall = tuple[str, str, dict[str, Any]]


def openai_message(content: str = "", calls: Sequence[ScriptedCall] = ()) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content or None}
    if calls:
        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
            for call_id, name, arguments in calls
        ]
    return {
        "choices": [{"message": message, "finish_reason": "tool_calls" if calls else "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def openai_stream(content: str = "", calls: Sequence[ScriptedCall] = ()) -> list[str]:
    chunks: list[str] = []
    if content:
        chunks.append(json.dumps({"choices": [{"delta": {"content": content}}]}))
    for index, (call_id, name, arguments) in enumerate(calls):
        delta = {
            "tool_calls": [
                {
                    "index": index,
                    "id": call_id,
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ]
        }
        chunks.append(json.dumps({"choices": [{"delta": delta}]}))
    reason = "tool_calls" if calls else "stop"
    chunks.append(json.dumps({"choices": [{"delta": {}, "finish_reason": reason}]}))
    chunks.append(
        json.dumps({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    )
    chunks.append("[DONE]")
    return chunks


def scripted_router(
    *,
    json_responses: Sequence[dict[str, Any] | Exception] = (),
    streams: Sequence[Sequence[str | Exception] | Exception] = (),
    model: str = "groq/test",
) -> tuple[Router, FakeTransport]:
    transport = FakeTransport(json_responses, streams)
    provider = OpenAICompatProvider(
        BUILTIN_PROVIDERS["groq"], "k", transport, RetryPolicy(max_retries=0), RecordingSleep()
    )
    return Router({"groq": provider}, [ModelRef.parse(model)]), transport

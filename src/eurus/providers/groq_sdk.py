from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

from eurus.catalog import ProviderSpec
from eurus.errors import ProviderError
from eurus.providers.base import RetryPolicy, Sleep
from eurus.providers.openai_compat import OpenAICompatProvider
from eurus.providers.sdk import dump_model, error_from_sdk, import_sdk
from eurus.providers.transport import HttpTransport

DEFAULT_ROOT = "https://api.groq.com"
OPENAI_SUFFIX = "/openai/v1"


async def _close_resource(resource: Any) -> None:
    closer = getattr(resource, "close", None)
    if callable(closer):
        result = closer()
        if hasattr(result, "__await__"):
            await result


class GroqSdkProvider(OpenAICompatProvider):
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
        self._sdk: Any = None if client is not None else import_sdk("groq", "groq")

    def _get_client(self) -> Any:
        if self._client is None:
            options: dict[str, Any] = {
                "api_key": self._key(),
                "timeout": self._timeout,
                "max_retries": 0,
            }
            root = self.spec.base_url.rstrip("/").removesuffix(OPENAI_SUFFIX)
            if root != DEFAULT_ROOT:
                options["base_url"] = root
            headers = self._extra_headers()
            if headers:
                options["default_headers"] = headers
            self._client = self._sdk.AsyncGroq(**options)
        return self._client

    async def _post(
        self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        client = self._get_client()
        try:
            completion = await client.chat.completions.create(**payload)
        except ProviderError:
            raise
        except Exception as exc:
            raise error_from_sdk(self.name, exc) from exc
        return dump_model(completion)

    async def _sse(
        self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any]
    ) -> AsyncIterator[str]:
        client = self._get_client()
        try:
            stream = await client.chat.completions.create(**payload)
            try:
                async for chunk in stream:
                    yield json.dumps(dump_model(chunk), default=str)
            finally:
                await _close_resource(stream)
        except ProviderError:
            raise
        except Exception as exc:
            raise error_from_sdk(self.name, exc) from exc

    async def aclose(self) -> None:
        await _close_resource(self._client)

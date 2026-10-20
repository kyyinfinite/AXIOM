from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

from eurus import __version__
from eurus.providers.transport import (
    HttpError,
    TransportError,
    is_cloudflare_block,
    parse_sse_line,
    strip_markup,
)

USER_AGENT = f"eurus/{__version__}"
KEPT_HEADERS = ("cf-ray", "server", "content-type")
NON_SSE_LIMIT = 300


def user_agent_candidates(headers: Mapping[str, str]) -> tuple[str | None, ...]:
    if any(key.lower() == "user-agent" for key in headers):
        return (None,)
    return (USER_AGENT, f"Mozilla/5.0 (compatible; {USER_AGENT})")


def with_user_agent(headers: Mapping[str, str], agent: str | None) -> dict[str, str]:
    merged = dict(headers)
    if agent is not None:
        merged["User-Agent"] = agent
    return merged


def _retry_after(headers: httpx.Headers) -> float | None:
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _kept(headers: httpx.Headers) -> dict[str, str]:
    return {name: headers[name] for name in KEPT_HEADERS if name in headers}


def _http_error(response: httpx.Response, body: str) -> HttpError:
    return HttpError(
        response.status_code, body, _retry_after(response.headers), _kept(response.headers)
    )


def _validate_header_values(headers: Mapping[str, str]) -> None:
    """Reject invalid control characters before custom transports can bypass HTTP validation."""
    for name, value in headers.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise TransportError("invalid request header; header names and values must be text")
        if any(ord(char) < 32 and char != "\t" or ord(char) == 127 for char in name + value):
            raise TransportError(
                "invalid request header; check the API key and custom headers for "
                "stray whitespace or control characters"
            )


def _transport_error(exc: httpx.HTTPError) -> TransportError:
    if isinstance(exc, httpx.TimeoutException):
        return TransportError("request timed out")
    if isinstance(exc, httpx.LocalProtocolError):
        return TransportError(
            "invalid request header; check the API key and custom headers for "
            "stray whitespace, newlines or non-ASCII characters"
        )
    return TransportError(f"network error: {type(exc).__name__}")


class HttpxTransport:
    def __init__(self, timeout: float = 120.0, client: httpx.AsyncClient | None = None) -> None:
        if client is None:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout, connect=15.0),
                follow_redirects=False,
            )
        self._client = client

    async def post_json(
        self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        candidates = user_agent_candidates(headers)
        for index, agent in enumerate(candidates):
            request_headers = with_user_agent(headers, agent)
            _validate_header_values(request_headers)
            try:
                # Validate headers before handing the request to a transport. Mock/custom
                # transports may skip httpcore's normal protocol validation.
                httpx.Request("POST", url, headers=request_headers, json=dict(payload))
                response = await self._client.post(url, headers=request_headers, json=dict(payload))
            except (httpx.HTTPError, ValueError) as exc:
                if isinstance(exc, httpx.HTTPError):
                    raise _transport_error(exc) from exc
                raise TransportError(
                    "invalid request header or request data; check custom headers"
                ) from exc
            if response.status_code < 400:
                return self._json(response)
            error = _http_error(response, response.text)
            if is_cloudflare_block(error) and index + 1 < len(candidates):
                continue
            raise error
        raise TransportError("no request was sent")

    @staticmethod
    def _json(response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise TransportError("response is not valid JSON") from exc
        if not isinstance(data, dict):
            raise TransportError("response JSON is not an object")
        return data

    async def stream_sse(
        self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any]
    ) -> AsyncIterator[str]:
        candidates = user_agent_candidates(headers)
        base = {"Accept": "text/event-stream", **headers}
        for index, agent in enumerate(candidates):
            blocked: HttpError | None = None
            seen = False
            stray: list[str] = []
            try:
                request_headers = with_user_agent(base, agent)
                _validate_header_values(request_headers)
                httpx.Request("POST", url, headers=request_headers, json=dict(payload))
                async with self._client.stream(
                    "POST", url, headers=request_headers, json=dict(payload)
                ) as response:
                    if response.status_code >= 400:
                        raw = (await response.aread()).decode("utf-8", errors="replace")
                        error = _http_error(response, raw)
                        if not (is_cloudflare_block(error) and index + 1 < len(candidates)):
                            raise error
                        blocked = error
                    else:
                        async for line in response.aiter_lines():
                            data = parse_sse_line(line)
                            if data is not None:
                                seen = True
                                yield data
                            elif line.strip() and not seen and len(stray) < NON_SSE_LIMIT:
                                stray.append(line.strip())
            except (httpx.HTTPError, ValueError) as exc:
                if isinstance(exc, httpx.HTTPError):
                    raise _transport_error(exc) from exc
                raise TransportError(
                    "invalid request header or request data; check custom headers"
                ) from exc
            if blocked is not None:
                continue
            if not seen and stray:
                snippet = strip_markup(" ".join(stray))[:NON_SSE_LIMIT]
                raise TransportError(f"server did not return an event stream: {snippet}")
            return

    async def aclose(self) -> None:
        await self._client.aclose()

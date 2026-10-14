from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from axiom.catalog import resolve_api_key
from axiom.config import Config
from axiom.errors import AccessDeniedError, AxiomError
from axiom.logs import redact
from axiom.models import Message, ModelRef
from axiom.providers.base import ChatRequest, RetryPolicy
from axiom.providers.factory import build_provider
from axiom.providers.httpx_transport import HttpxTransport
from axiom.providers.transport import HttpTransport

PROBE_PROMPT = "Reply with the single word: ok"
PROBE_TOKENS = 16


@dataclass(frozen=True, slots=True)
class ProbeResult:
    model: str
    ok: bool
    detail: str
    seconds: float


async def probe_model(
    config: Config,
    ref: ModelRef,
    *,
    transport: HttpTransport | None = None,
    environ: Mapping[str, str] | None = None,
) -> ProbeResult:
    started = time.monotonic()
    spec = config.providers[ref.provider]
    key = resolve_api_key(spec, environ)
    shared: HttpTransport = (
        HttpxTransport(timeout=config.network.timeout_seconds) if transport is None else transport
    )
    secrets = (key,) if key else ()
    try:
        provider = build_provider(
            spec, key, shared, RetryPolicy(max_retries=0), config.network.timeout_seconds
        )
        try:
            request = ChatRequest(
                model=ref.model,
                messages=(Message.user(PROBE_PROMPT),),
                max_tokens=PROBE_TOKENS,
            )
            response = await provider.chat(request)
        finally:
            await provider.aclose()
        reply = " ".join(response.message.content.split())[:40]
        ok, detail = True, f"reply={reply!r} in={response.usage.input_tokens}"
    except AccessDeniedError as exc:
        ok, detail = False, f"{exc.reason}: {redact(str(exc), secrets)}"
    except AxiomError as exc:
        ok, detail = False, redact(str(exc), secrets)
    finally:
        if transport is None:
            await shared.aclose()
    return ProbeResult(str(ref), ok, detail, time.monotonic() - started)


async def probe_all(
    config: Config,
    refs: Sequence[ModelRef],
    *,
    transport: HttpTransport | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[ProbeResult]:
    return [await probe_model(config, ref, transport=transport, environ=environ) for ref in refs]

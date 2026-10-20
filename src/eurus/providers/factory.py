from __future__ import annotations

from collections.abc import Mapping

from eurus.catalog import ProviderSpec, resolve_api_key
from eurus.config import Config
from eurus.errors import ConfigError, ModelRefError, PolicyError, ProviderError
from eurus.models import ModelRef
from eurus.providers.anthropic import AnthropicProvider
from eurus.providers.base import Provider, RetryPolicy
from eurus.providers.gemini import GeminiProvider
from eurus.providers.genai_sdk import GenaiSdkProvider
from eurus.providers.groq_sdk import GroqSdkProvider
from eurus.providers.httpx_transport import HttpxTransport
from eurus.providers.openai_compat import OpenAICompatProvider
from eurus.providers.policy import is_local_url
from eurus.providers.router import Router
from eurus.providers.transport import HttpTransport

PROVIDER_CLASSES: dict[str, type[Provider]] = {
    "openai": OpenAICompatProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}


def build_provider(
    spec: ProviderSpec,
    api_key: str | None,
    transport: HttpTransport,
    retry: RetryPolicy | None = None,
    timeout: float = 120.0,
) -> Provider:
    if spec.protocol == "groq-sdk":
        return GroqSdkProvider(spec, api_key, transport, retry, timeout=timeout)
    if spec.protocol == "genai-sdk":
        return GenaiSdkProvider(spec, api_key, transport, retry, timeout=timeout)
    provider_class = PROVIDER_CLASSES.get(spec.protocol)
    if provider_class is None:
        raise ProviderError(f"Unsupported provider protocol: {spec.protocol}", provider=spec.name)
    return provider_class(spec, api_key, transport, retry)


def resolve_chain(config: Config, model: str | None = None) -> list[ModelRef]:
    primary = (model or config.agent.model).strip()
    if not primary:
        raise ConfigError("no model configured: set agent.model, EURUS_MODEL or pass --model")
    chain: list[ModelRef] = []
    for text in (primary, *config.agent.fallback):
        try:
            ref = ModelRef.parse(text)
        except ModelRefError as exc:
            raise ConfigError(str(exc)) from exc
        if ref.provider not in config.providers:
            raise ConfigError(f"unknown provider '{ref.provider}' in model '{text}'")
        if ref not in chain:
            chain.append(ref)
    if config.network.local_only:
        chain = [ref for ref in chain if is_local_url(config.providers[ref.provider].base_url)]
        if not chain:
            raise PolicyError(
                "network.local_only is enabled and no configured model uses a local endpoint"
            )
    return chain


def resolve_auxiliary_chain(config: Config) -> list[ModelRef]:
    primary = config.auxiliary.model.strip()
    if not primary:
        return []
    chain: list[ModelRef] = []
    for text in (primary, *config.auxiliary.fallback):
        try:
            ref = ModelRef.parse(text)
        except ModelRefError as exc:
            raise ConfigError(str(exc)) from exc
        if ref.provider not in config.providers:
            raise ConfigError(f"unknown provider '{ref.provider}' in auxiliary model '{text}'")
        if ref not in chain:
            chain.append(ref)
    if config.network.local_only:
        chain = [ref for ref in chain if is_local_url(config.providers[ref.provider].base_url)]
        if not chain:
            raise PolicyError(
                "network.local_only is enabled and no auxiliary model uses a local endpoint"
            )
    return chain


def build_auxiliary_router(
    config: Config,
    *,
    transport: HttpTransport | None = None,
    environ: Mapping[str, str] | None = None,
) -> Router | None:
    chain = resolve_auxiliary_chain(config)
    if not chain:
        return None
    owned = transport is None
    shared: HttpTransport = (
        HttpxTransport(timeout=config.network.timeout_seconds) if transport is None else transport
    )
    retry = RetryPolicy(max_retries=config.network.max_retries)
    providers: dict[str, Provider] = {}
    for ref in chain:
        if ref.provider in providers:
            continue
        spec = config.providers[ref.provider]
        providers[ref.provider] = build_provider(
            spec,
            resolve_api_key(spec, environ),
            shared,
            retry,
            config.network.timeout_seconds,
        )
    return Router(
        providers,
        chain,
        temperature=config.agent.temperature,
        transport=shared if owned else None,
    )


def build_router(
    config: Config,
    *,
    model: str | None = None,
    transport: HttpTransport | None = None,
    environ: Mapping[str, str] | None = None,
) -> Router:
    chain = resolve_chain(config, model)
    owned = transport is None
    shared: HttpTransport = (
        HttpxTransport(timeout=config.network.timeout_seconds) if transport is None else transport
    )
    retry = RetryPolicy(max_retries=config.network.max_retries)
    providers: dict[str, Provider] = {}
    for ref in chain:
        if ref.provider in providers:
            continue
        spec = config.providers[ref.provider]
        providers[ref.provider] = build_provider(
            spec,
            resolve_api_key(spec, environ),
            shared,
            retry,
            config.network.timeout_seconds,
        )
    return Router(
        providers,
        chain,
        temperature=config.agent.temperature,
        transport=shared if owned else None,
    )

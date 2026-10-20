from __future__ import annotations

from dataclasses import dataclass

from eurus.catalog import resolve_api_key
from eurus.config import Config
from eurus.models import ModelRef


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    provider: str
    model: str
    credential: str


def describe(config: Config, model: str | None = None) -> ProviderInfo:
    text = config.agent.model if model is None else model
    if not text:
        return ProviderInfo("not configured", "not configured", "not configured")
    try:
        ref = ModelRef.parse(text)
    except ValueError:
        return ProviderInfo("invalid", text, "invalid model")
    spec = config.providers.get(ref.provider)
    if spec is None:
        return ProviderInfo(ref.provider, text, "unknown provider")
    if not spec.api_key_env:
        return ProviderInfo(ref.provider, text, "no key needed")
    return ProviderInfo(ref.provider, text, "key set" if resolve_api_key(spec) else "no key")


def credential_note(credential: str) -> str:
    if credential == "key set":
        return "key set, not verified"
    if credential == "no key":
        return "no key found"
    if credential == "no key needed":
        return "no key needed"
    return credential


def config_rows(config: Config) -> list[tuple[str, str]]:
    agent = config.agent
    return [
        ("model", agent.model or "not configured"),
        ("fallback", ", ".join(agent.fallback) or "none"),
        ("skills", ", ".join(agent.skills) or "none"),
        ("max steps", str(agent.max_steps)),
        ("context tokens", str(agent.context_tokens)),
        ("permission mode", config.permissions.mode),
        ("workspace", config.permissions.workspace),
        ("sandbox", f"{config.sandbox.backend} / {config.sandbox.runtime}"),
        ("sandbox network", "on" if config.sandbox.network else "off"),
        ("local only", "yes" if config.network.local_only else "no"),
        ("learning", config.learning.mode),
    ]


def shell_label(config: Config) -> str:
    if config.sandbox.backend == "host":
        return "host, not isolated (asks approval)"
    return f"{config.sandbox.runtime} container"

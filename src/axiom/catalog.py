from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace

PROTOCOLS: tuple[str, ...] = ("openai", "anthropic", "gemini", "groq-sdk", "genai-sdk")
QUOTES = "'\""
BEARER_PREFIX = "bearer "


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    name: str
    protocol: str
    base_url: str
    api_key_env: str = ""
    user_agent: str = ""
    extra_headers: tuple[tuple[str, str], ...] = ()


def _spec(name: str, protocol: str, base_url: str, api_key_env: str = "") -> ProviderSpec:
    return ProviderSpec(name, protocol, base_url, api_key_env)


BUILTIN_PROVIDERS: dict[str, ProviderSpec] = {
    spec.name: spec
    for spec in (
        _spec("openai", "openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
        _spec("anthropic", "anthropic", "https://api.anthropic.com", "ANTHROPIC_API_KEY"),
        _spec(
            "gemini",
            "gemini",
            "https://generativelanguage.googleapis.com",
            "GEMINI_API_KEY",
        ),
        _spec(
            "gemini-sdk",
            "genai-sdk",
            "https://generativelanguage.googleapis.com",
            "GEMINI_API_KEY",
        ),
        _spec("groq", "openai", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
        _spec("groq-sdk", "groq-sdk", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
        _spec("nvidia", "openai", "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
        _spec("openrouter", "openai", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
        _spec("deepseek", "openai", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
        _spec("mistral", "openai", "https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
        _spec("together", "openai", "https://api.together.xyz/v1", "TOGETHER_API_KEY"),
        _spec("xai", "openai", "https://api.x.ai/v1", "XAI_API_KEY"),
        _spec("ollama", "openai", "http://127.0.0.1:11434/v1"),
    )
}


def builtin_providers(environ: Mapping[str, str] | None = None) -> dict[str, ProviderSpec]:
    env = os.environ if environ is None else environ
    providers = dict(BUILTIN_PROVIDERS)
    ollama_base = env.get("OLLAMA_BASE_URL", "").strip().rstrip("/")
    if ollama_base:
        providers["ollama"] = replace(providers["ollama"], base_url=f"{ollama_base}/v1")
    nvidia_base = env.get("NVIDIA_BASE_URL", "").strip().rstrip("/")
    if nvidia_base:
        providers["nvidia"] = replace(providers["nvidia"], base_url=nvidia_base)
    return providers


def clean_secret(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in QUOTES:
        text = text[1:-1].strip()
    if text.lower().startswith(BEARER_PREFIX):
        text = text[len(BEARER_PREFIX) :].strip()
    return "".join(text.split())


def resolve_api_key(spec: ProviderSpec, environ: Mapping[str, str] | None = None) -> str | None:
    if not spec.api_key_env:
        return None
    env = os.environ if environ is None else environ
    value = clean_secret(env.get(spec.api_key_env, ""))
    return value or None


def is_ready(spec: ProviderSpec, environ: Mapping[str, str] | None = None) -> bool:
    return not spec.api_key_env or resolve_api_key(spec, environ) is not None

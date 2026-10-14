from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from axiom import paths
from axiom.catalog import PROTOCOLS, ProviderSpec, builtin_providers
from axiom.errors import ConfigError, ModelRefError
from axiom.models import ModelRef

PERMISSION_MODES: tuple[str, ...] = ("ask", "auto-edit", "yolo")
LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR")
TOP_LEVEL_KEYS = frozenset(
    {"agent", "permissions", "logging", "network", "tools", "providers", "sandbox"}
)
PROVIDER_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
HEADER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*$")
TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
FALSE_VALUES = frozenset({"0", "false", "no", "off"})

ENV_OVERRIDES: tuple[tuple[str, str, str, type[Any]], ...] = (
    ("AXIOM_MODEL", "agent", "model", str),
    ("AXIOM_FALLBACK", "agent", "fallback", list),
    ("AXIOM_MAX_STEPS", "agent", "max_steps", int),
    ("AXIOM_TEMPERATURE", "agent", "temperature", float),
    ("AXIOM_PERMISSION_MODE", "permissions", "mode", str),
    ("AXIOM_WORKSPACE", "permissions", "workspace", str),
    ("AXIOM_LOG_LEVEL", "logging", "level", str),
    ("AXIOM_LOG_FILE", "logging", "file", str),
    ("AXIOM_LOG_JSON", "logging", "json", bool),
    ("AXIOM_TIMEOUT", "network", "timeout_seconds", float),
    ("AXIOM_MAX_RETRIES", "network", "max_retries", int),
    ("AXIOM_LOCAL_ONLY", "network", "local_only", bool),
)

SAMPLE_CONFIG = """[agent]
model = "groq/llama-3.3-70b-versatile"
fallback = ["openai/gpt-4o-mini"]
max_steps = 40
temperature = 0.2
system_prompt = ""

[permissions]
mode = "ask"
workspace = "."

[logging]
level = "INFO"
file = ""
json = false

[network]
timeout_seconds = 120.0
max_retries = 3
local_only = false

[tools]
max_output_chars = 20000
timeout_seconds = 120.0
max_parallel_tools = 4

[sandbox]
backend = "container"
runtime = "docker"
image = "python:3.12-alpine"
required = true
network = false
memory_mb = 1024
cpu_limit = 1.0
pids_limit = 128
"""


@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str = ""
    fallback: tuple[str, ...] = ()
    max_steps: int = 40
    temperature: float = 0.2
    system_prompt: str = ""


@dataclass(frozen=True, slots=True)
class PermissionConfig:
    mode: str = "ask"
    workspace: str = "."


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    level: str = "INFO"
    file: str = ""
    json: bool = False


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    timeout_seconds: float = 120.0
    max_retries: int = 3
    local_only: bool = False


@dataclass(frozen=True, slots=True)
class ToolsConfig:
    max_output_chars: int = 20000
    timeout_seconds: float = 120.0
    max_parallel_tools: int = 4


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    backend: str = "container"
    runtime: str = "docker"
    image: str = "python:3.12-alpine"
    required: bool = True
    network: bool = False
    memory_mb: int = 1024
    cpu_limit: float = 1.0
    pids_limit: int = 128


@dataclass(frozen=True, slots=True)
class Config:
    agent: AgentConfig = field(default_factory=AgentConfig)
    permissions: PermissionConfig = field(default_factory=PermissionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    providers: dict[str, ProviderSpec] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def deep_merge(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in extra.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(current, value)
        elif isinstance(value, Mapping):
            result[key] = deep_merge({}, value)
        else:
            result[key] = value
    return result


def read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc


def _coerce_env(name: str, text: str, kind: type[Any]) -> Any:
    cleaned = text.strip()
    if kind is list:
        return [part.strip() for part in cleaned.split(",") if part.strip()]
    if kind is bool:
        lowered = cleaned.lower()
        if lowered in TRUE_VALUES:
            return True
        if lowered in FALSE_VALUES:
            return False
        raise ConfigError(f"{name} has invalid value: {text!r}")
    try:
        return kind(cleaned)
    except ValueError as exc:
        raise ConfigError(f"{name} has invalid value: {text!r}") from exc


def env_overrides(environ: Mapping[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, section, key, kind in ENV_OVERRIDES:
        text = environ.get(name, "")
        if not text.strip():
            continue
        result.setdefault(section, {})[key] = _coerce_env(name, text, kind)
    return result


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"[{name}] must be a table")
    return value


def _check_keys(section: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ConfigError(f"unknown key(s) in [{where}]: {', '.join(unknown)}")


def _get(section: Mapping[str, Any], key: str, kind: type[Any], default: Any, where: str) -> Any:
    if key not in section:
        return default
    value = section[key]
    is_bool = isinstance(value, bool)
    if kind is float and isinstance(value, int) and not is_bool:
        return float(value)
    if kind is not bool and is_bool:
        raise ConfigError(f"{where}.{key} must be {kind.__name__}")
    if not isinstance(value, kind):
        raise ConfigError(f"{where}.{key} must be {kind.__name__}")
    return value


def _str_list(section: Mapping[str, Any], key: str, where: str) -> list[str]:
    value = section.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{where}.{key} must be a list of strings")
    return [item.strip() for item in value if item.strip()]


def _check_model(text: str, providers: Mapping[str, ProviderSpec], where: str) -> None:
    try:
        ref = ModelRef.parse(text)
    except ModelRefError as exc:
        raise ConfigError(f"{where}: {exc}") from exc
    if ref.provider not in providers:
        raise ConfigError(f"{where}: unknown provider '{ref.provider}'")


def _header_value(value: str, where: str) -> str:
    if "\r" in value or "\n" in value:
        raise ConfigError(f"{where} must not contain line breaks")
    return value


def _build_headers(
    entry: Mapping[str, Any], existing: ProviderSpec | None, where: str
) -> tuple[tuple[str, str], ...]:
    if "headers" not in entry:
        return existing.extra_headers if existing else ()
    table = entry["headers"]
    if not isinstance(table, Mapping):
        raise ConfigError(f"{where}.headers must be a table")
    result: list[tuple[str, str]] = []
    for key, value in table.items():
        if not HEADER_NAME.match(str(key)) or not isinstance(value, str):
            raise ConfigError(f"{where}.headers has an invalid entry: {key!r}")
        result.append((str(key), _header_value(value.strip(), f"{where}.headers.{key}")))
    return tuple(result)


def _build_providers(raw: Mapping[str, Any], environ: Mapping[str, str]) -> dict[str, ProviderSpec]:
    providers = builtin_providers(environ)
    for name, entry in _section(raw, "providers").items():
        where = f"providers.{name}"
        if not PROVIDER_NAME.match(name):
            raise ConfigError(f"invalid provider name: {name!r}")
        if not isinstance(entry, Mapping):
            raise ConfigError(f"[{where}] must be a table")
        _check_keys(entry, {"protocol", "base_url", "api_key_env", "user_agent", "headers"}, where)
        existing = providers.get(name)
        protocol = _get(entry, "protocol", str, existing.protocol if existing else "", where)
        base_url = _get(entry, "base_url", str, existing.base_url if existing else "", where)
        api_key_env = _get(
            entry, "api_key_env", str, existing.api_key_env if existing else "", where
        )
        if protocol not in PROTOCOLS:
            raise ConfigError(f"{where}.protocol must be one of: {', '.join(PROTOCOLS)}")
        if not base_url.startswith(("http://", "https://")):
            raise ConfigError(f"{where}.base_url must start with http:// or https://")
        user_agent = _get(entry, "user_agent", str, existing.user_agent if existing else "", where)
        providers[name] = ProviderSpec(
            name=name,
            protocol=protocol,
            base_url=base_url.rstrip("/"),
            api_key_env=api_key_env.strip(),
            user_agent=_header_value(user_agent.strip(), f"{where}.user_agent"),
            extra_headers=_build_headers(entry, existing, where),
        )
    return providers


def _build_agent(raw: Mapping[str, Any], providers: Mapping[str, ProviderSpec]) -> AgentConfig:
    section = _section(raw, "agent")
    _check_keys(
        section, {"model", "fallback", "max_steps", "temperature", "system_prompt"}, "agent"
    )
    model = _get(section, "model", str, "", "agent").strip()
    fallback = tuple(_str_list(section, "fallback", "agent"))
    max_steps = _get(section, "max_steps", int, 40, "agent")
    temperature = _get(section, "temperature", float, 0.2, "agent")
    system_prompt = _get(section, "system_prompt", str, "", "agent").strip()
    if not 1 <= max_steps <= 1000:
        raise ConfigError("agent.max_steps must be between 1 and 1000")
    if not 0.0 <= temperature <= 2.0:
        raise ConfigError("agent.temperature must be between 0 and 2")
    if model:
        _check_model(model, providers, "agent.model")
    for item in fallback:
        _check_model(item, providers, "agent.fallback")
    return AgentConfig(
        model=model,
        fallback=fallback,
        max_steps=max_steps,
        temperature=temperature,
        system_prompt=system_prompt,
    )


def _build_permissions(raw: Mapping[str, Any]) -> PermissionConfig:
    section = _section(raw, "permissions")
    _check_keys(section, {"mode", "workspace"}, "permissions")
    mode = _get(section, "mode", str, "ask", "permissions").strip()
    workspace = _get(section, "workspace", str, ".", "permissions").strip() or "."
    if mode not in PERMISSION_MODES:
        raise ConfigError(f"permissions.mode must be one of: {', '.join(PERMISSION_MODES)}")
    return PermissionConfig(mode=mode, workspace=workspace)


def _build_logging(raw: Mapping[str, Any]) -> LoggingConfig:
    section = _section(raw, "logging")
    _check_keys(section, {"level", "file", "json"}, "logging")
    level = _get(section, "level", str, "INFO", "logging").strip().upper()
    file = _get(section, "file", str, "", "logging").strip()
    as_json = _get(section, "json", bool, False, "logging")
    if level not in LOG_LEVELS:
        raise ConfigError(f"logging.level must be one of: {', '.join(LOG_LEVELS)}")
    return LoggingConfig(level=level, file=file, json=as_json)


def _build_network(raw: Mapping[str, Any]) -> NetworkConfig:
    section = _section(raw, "network")
    _check_keys(section, {"timeout_seconds", "max_retries", "local_only"}, "network")
    timeout = _get(section, "timeout_seconds", float, 120.0, "network")
    retries = _get(section, "max_retries", int, 3, "network")
    local_only = _get(section, "local_only", bool, False, "network")
    if not 1.0 <= timeout <= 600.0:
        raise ConfigError("network.timeout_seconds must be between 1 and 600")
    if not 0 <= retries <= 10:
        raise ConfigError("network.max_retries must be between 0 and 10")
    return NetworkConfig(timeout_seconds=timeout, max_retries=retries, local_only=local_only)


def _build_tools(raw: Mapping[str, Any]) -> ToolsConfig:
    section = _section(raw, "tools")
    _check_keys(section, {"max_output_chars", "timeout_seconds", "max_parallel_tools"}, "tools")
    limit = _get(section, "max_output_chars", int, 20000, "tools")
    timeout = _get(section, "timeout_seconds", float, 120.0, "tools")
    parallel = _get(section, "max_parallel_tools", int, 4, "tools")
    if not 500 <= limit <= 500000:
        raise ConfigError("tools.max_output_chars must be between 500 and 500000")
    if not 1.0 <= timeout <= 3600.0:
        raise ConfigError("tools.timeout_seconds must be between 1 and 3600")
    if not 1 <= parallel <= 32:
        raise ConfigError("tools.max_parallel_tools must be between 1 and 32")
    return ToolsConfig(max_output_chars=limit, timeout_seconds=timeout, max_parallel_tools=parallel)


def _build_sandbox(raw: Mapping[str, Any]) -> SandboxConfig:
    section = _section(raw, "sandbox")
    _check_keys(
        section,
        {
            "backend",
            "runtime",
            "image",
            "required",
            "network",
            "memory_mb",
            "cpu_limit",
            "pids_limit",
        },
        "sandbox",
    )
    backend = _get(section, "backend", str, "container", "sandbox").strip().lower()
    runtime = _get(section, "runtime", str, "docker", "sandbox").strip()
    image = _get(section, "image", str, "python:3.12-alpine", "sandbox").strip()
    required = _get(section, "required", bool, True, "sandbox")
    network = _get(section, "network", bool, False, "sandbox")
    memory_mb = _get(section, "memory_mb", int, 1024, "sandbox")
    cpu_limit = _get(section, "cpu_limit", float, 1.0, "sandbox")
    pids_limit = _get(section, "pids_limit", int, 128, "sandbox")
    if backend != "container":
        raise ConfigError("sandbox.backend currently supports only 'container'")
    if not runtime or not image:
        raise ConfigError("sandbox.runtime and sandbox.image must not be empty")
    if not 128 <= memory_mb <= 65536:
        raise ConfigError("sandbox.memory_mb must be between 128 and 65536")
    if not 0.1 <= cpu_limit <= 64:
        raise ConfigError("sandbox.cpu_limit must be between 0.1 and 64")
    if not 16 <= pids_limit <= 4096:
        raise ConfigError("sandbox.pids_limit must be between 16 and 4096")
    return SandboxConfig(
        backend, runtime, image, required, network, memory_mb, cpu_limit, pids_limit
    )


def build_config(raw: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> Config:
    env = os.environ if environ is None else environ
    unknown = sorted(set(raw) - TOP_LEVEL_KEYS)
    if unknown:
        raise ConfigError(f"unknown section(s): {', '.join(unknown)}")
    providers = _build_providers(raw, env)
    return Config(
        agent=_build_agent(raw, providers),
        permissions=_build_permissions(raw),
        logging=_build_logging(raw),
        network=_build_network(raw),
        tools=_build_tools(raw),
        sandbox=_build_sandbox(raw),
        providers=providers,
    )


def load_config(
    *,
    config_path: Path | None = None,
    cwd: Path | None = None,
    environ: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    env = os.environ if environ is None else environ
    raw: dict[str, Any] = {}
    if config_path is not None:
        if not config_path.is_file():
            raise ConfigError(f"config file not found: {config_path}")
        raw = deep_merge(raw, read_toml(config_path))
    else:
        user_file = paths.user_config_path(env)
        if user_file.is_file():
            raw = deep_merge(raw, read_toml(user_file))
        project_file = paths.project_config_path(cwd or Path.cwd())
        if project_file is not None:
            raw = deep_merge(raw, read_toml(project_file))
    raw = deep_merge(raw, env_overrides(env))
    if overrides:
        raw = deep_merge(raw, overrides)
    return build_config(raw, env)

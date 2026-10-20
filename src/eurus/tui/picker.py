from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from eurus.catalog import is_ready
from eurus.config import Config
from eurus.models import ModelRef
from eurus.providers.discovery import ModelInfo
from eurus.providers.policy import is_local_url

ChoiceKind = Literal["quit", "next", "prev", "back", "refresh", "select", "text"]

MIN_PAGE = 5
MAX_PAGE = 12
CHROME_ROWS = 12


@dataclass(frozen=True, slots=True)
class ProviderRow:
    name: str
    ready: bool
    active: bool
    note: str
    env: str = ""


@dataclass(frozen=True, slots=True)
class Choice:
    kind: ChoiceKind
    index: int = 0
    text: str = ""


def active_ref(config: Config) -> ModelRef | None:
    try:
        return ModelRef.parse(config.agent.model)
    except ValueError:
        return None


def provider_rows(config: Config, environ: Mapping[str, str] | None = None) -> list[ProviderRow]:
    current = active_ref(config)
    rows: list[ProviderRow] = []
    for name, spec in config.providers.items():
        blocked = config.network.local_only and not is_local_url(spec.base_url)
        ready = is_ready(spec, environ) and not blocked
        if blocked:
            note = "blocked by local_only"
        elif ready:
            note = "ready" if spec.api_key_env else "ready (local)"
        else:
            note = spec.api_key_env
        active = current is not None and current.provider == name
        rows.append(ProviderRow(name, ready, active, note, spec.api_key_env))
    rows.sort(key=lambda row: (not row.active, not row.ready, row.name))
    return rows


def page_size(height: int) -> int:
    return max(MIN_PAGE, min(MAX_PAGE, height - CHROME_ROWS))


def page_bounds(total: int, page: int, size: int) -> tuple[int, int, int, int]:
    pages = max((total + size - 1) // size, 1)
    page = min(max(page, 0), pages - 1)
    start = page * size
    return page, pages, start, min(start + size, total)


def filter_models(models: Sequence[ModelInfo], query: str) -> list[ModelInfo]:
    needle = query.strip().lower()
    if not needle:
        return list(models)
    return [m for m in models if needle in m.id.lower() or needle in m.label.lower()]


def with_active_first(models: Sequence[ModelInfo], active_id: str) -> list[ModelInfo]:
    if not active_id:
        return list(models)
    head = [m for m in models if m.id == active_id]
    return head + [m for m in models if m.id != active_id]


def parse_choice(text: str, visible: int) -> Choice:
    raw = text.strip()
    low = raw.lower()
    if low in ("", "q", "quit", "exit", "cancel"):
        return Choice("quit")
    if low in ("n", "next", ">"):
        return Choice("next")
    if low in ("p", "prev", "previous", "<"):
        return Choice("prev")
    if low in ("b", "back"):
        return Choice("back")
    if low in ("r", "refresh"):
        return Choice("refresh")
    if raw.isdigit():
        number = int(raw)
        if 1 <= number <= visible:
            return Choice("select", index=number - 1)
        return Choice("text", text=raw)
    return Choice("text", text=raw)

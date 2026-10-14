from __future__ import annotations

import re
from collections.abc import MutableMapping, Sequence
from pathlib import Path

KEY_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
EXTRA_ALLOWED = frozenset({"OLLAMA_BASE_URL", "NVIDIA_BASE_URL"})
QUOTES = "'\""


def is_allowed(name: str) -> bool:
    return name.endswith("_API_KEY") or name in EXTRA_ALLOWED


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in QUOTES:
        return value[1:-1]
    marker = value.find(" #")
    return value if marker == -1 else value[:marker].rstrip()


def parse_dotenv(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if separator and KEY_NAME.match(name):
            values[name] = _unquote(value.strip())
    return values


def load_dotenv(files: Sequence[Path], environ: MutableMapping[str, str]) -> list[Path]:
    loaded: list[Path] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError:
            continue
        for name, value in parse_dotenv(text).items():
            if is_allowed(name) and value.strip() and not environ.get(name, "").strip():
                environ[name] = value
        loaded.append(path)
    return loaded

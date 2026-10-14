from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from axiom.catalog import resolve_api_key
from axiom.config import Config, LoggingConfig

LOGGER_NAME = "axiom"
REDACTED = "[REDACTED]"
MIN_SECRET_LENGTH = 6

_PREFIXED_SECRETS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"gsk_[A-Za-z0-9]{8,}"),
    re.compile(r"nvapi-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"bearer\s+[A-Za-z0-9._\-]{8,}", re.IGNORECASE),
)
_ASSIGNMENT = re.compile(
    r"""(api[_-]?key|token|secret|password)(\s*[=:]\s*)[^\s,;"']+""",
    re.IGNORECASE,
)


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    for secret in secrets:
        if len(secret) >= MIN_SECRET_LENGTH:
            text = text.replace(secret, REDACTED)
    for pattern in _PREFIXED_SECRETS:
        text = pattern.sub(REDACTED, text)
    return _ASSIGNMENT.sub(lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}", text)


def collect_secrets(config: Config, environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    values = (resolve_api_key(spec, environ) for spec in config.providers.values())
    return tuple(value for value in values if value)


class RedactingFormatter(logging.Formatter):
    def __init__(self, fmt: str | None = None, secrets: Iterable[str] = ()) -> None:
        super().__init__(fmt)
        self._secrets = tuple(secrets)

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record), self._secrets)


class JsonFormatter(RedactingFormatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return redact(json.dumps(payload, ensure_ascii=False), self._secrets)


def shutdown_logging() -> None:
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def setup_logging(settings: LoggingConfig, *, secrets: Iterable[str] = ()) -> logging.Logger:
    shutdown_logging()
    secret_values = tuple(secrets)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, settings.level.upper()))
    logger.propagate = False

    console: logging.Handler
    if settings.json:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(JsonFormatter(secrets=secret_values))
    else:
        console = RichHandler(console=Console(stderr=True), show_path=False, markup=False)
        console.setFormatter(RedactingFormatter("%(message)s", secret_values))
    logger.addHandler(console)

    if settings.file:
        target = Path(settings.file).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target, encoding="utf-8")
        if settings.json:
            file_handler.setFormatter(JsonFormatter(secrets=secret_values))
        else:
            file_handler.setFormatter(
                RedactingFormatter("%(asctime)s %(levelname)s %(name)s %(message)s", secret_values)
            )
        logger.addHandler(file_handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}")

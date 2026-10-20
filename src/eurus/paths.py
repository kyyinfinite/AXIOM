from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

HOME_DIR_NAME = ".eurus"
CONFIG_FILE_NAME = "config.toml"
PROJECT_FILE_NAMES: tuple[str, ...] = ("eurus.toml", ".eurus/config.toml")


def home_dir(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    custom = env.get("EURUS_HOME", "").strip()
    if custom:
        return Path(custom).expanduser()
    return Path.home() / HOME_DIR_NAME


def user_config_path(environ: Mapping[str, str] | None = None) -> Path:
    return home_dir(environ) / CONFIG_FILE_NAME


def project_config_path(cwd: Path) -> Path | None:
    for name in PROJECT_FILE_NAMES:
        candidate = cwd / name
        if candidate.is_file():
            return candidate
    return None


def ensure_private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path

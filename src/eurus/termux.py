from __future__ import annotations

import os
from collections.abc import Mapping

TERMUX_PREFIX = "/data/data/com.termux"
TERMUX_ENV = ("ANDROID_ROOT", "ANDROID_DATA", "TERMUX_VERSION")


def is_termux(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    if env.get("TERMUX_VERSION"):
        return True
    return env.get("PREFIX", "").startswith(TERMUX_PREFIX)

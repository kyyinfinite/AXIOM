from __future__ import annotations

from collections.abc import Mapping

from eurus.catalog import ProviderSpec, resolve_api_key
from eurus.providers.base import Provider
from eurus.providers.factory import build_provider
from eurus.providers.httpx_transport import HttpxTransport


def create_provider(spec: ProviderSpec, environ: Mapping[str, str] | None = None) -> Provider:
    return build_provider(spec, resolve_api_key(spec, environ), HttpxTransport())

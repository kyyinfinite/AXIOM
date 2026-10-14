from __future__ import annotations

from collections.abc import Mapping

from axiom.catalog import ProviderSpec, resolve_api_key
from axiom.providers.base import Provider
from axiom.providers.factory import build_provider
from axiom.providers.httpx_transport import HttpxTransport


def create_provider(spec: ProviderSpec, environ: Mapping[str, str] | None = None) -> Provider:
    return build_provider(spec, resolve_api_key(spec, environ), HttpxTransport())

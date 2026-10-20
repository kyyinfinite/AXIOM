from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

CGNAT = ipaddress.ip_network("100.64.0.0/10")


def is_local_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    if host == "localhost" or host.endswith((".localhost", ".local")):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_link_local or address in CGNAT

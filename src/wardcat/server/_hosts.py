"""Loopback checks for bind addresses and ``Host`` headers.

Starlette's ``TrustedHostMiddleware`` splits the header on the first colon, so
``[::1]:8787`` never matches anything; this module parses bracketed IPv6.
"""

from __future__ import annotations

import ipaddress

_LOOPBACK_NAMES = frozenset({"localhost"})


def address_is_loopback(host: str) -> bool:
    """Whether a bind address is loopback. A name other than ``localhost`` is not."""
    host = host.strip().strip("[]")
    if host.lower() in _LOOPBACK_NAMES:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(address.is_loopback or (mapped is not None and mapped.is_loopback))


def host_is_loopback(header: str) -> bool:
    """Whether a ``Host`` header names a loopback address, with or without a port."""
    header = header.strip()
    if not header:
        return False
    if header.startswith("["):
        end = header.find("]")
        if end == -1:
            return False
        rest = header[end + 1 :]
        if rest and not (rest.startswith(":") and rest[1:].isdigit()):
            return False
        return address_is_loopback(header[1:end])
    name, sep, port = header.rpartition(":")
    if sep and port.isdigit() and ":" not in name:
        return address_is_loopback(name)
    return address_is_loopback(header)

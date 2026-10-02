"""The copy-paste ASGI middleware fails closed: an unscanned body never reaches the route.

Until 1.2.2 a body over ``max_body_bytes`` and a scan that raised were both
forwarded to the route as they came. People copy examples into production, so
the example now refuses both (413 and 503) unless told otherwise.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from wardcat import Action, Entity, Wardcat

_PATH = Path(__file__).resolve().parents[2] / "examples" / "asgi_middleware.py"
_spec = importlib.util.spec_from_file_location("asgi_middleware_example", _PATH)
assert _spec and _spec.loader
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)


class Route:
    """A downstream app that records what reached it."""

    def __init__(self) -> None:
        self.bodies: list[bytes] = []

    async def __call__(self, scope, receive, send) -> None:
        message = await receive()
        self.bodies.append(message.get("body", b""))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def call(middleware, body: bytes, content_type: bytes = b"text/plain"):
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "path": "/echo", "headers": [(b"content-type", content_type)]}
    asyncio.run(middleware(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    payload = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], payload


def guard() -> Wardcat:
    return Wardcat(salt="s").add_entity(Entity.CREDIT_CARD, Action.REDACT)


class BrokenGuard:
    async def scan_async(self, text):
        raise RuntimeError("model unavailable")


def test_a_clean_sanitized_body_reaches_the_route() -> None:
    route = Route()
    mw = example.WardcatMiddleware(route, guard=guard())
    status, _ = call(mw, b"card 4111 1111 1111 1111")
    assert status == 200
    assert route.bodies == [b"card [CREDIT_CARD]"]


def test_an_oversized_body_is_refused_not_forwarded() -> None:
    route = Route()
    mw = example.WardcatMiddleware(route, guard=guard(), max_body_bytes=10)
    status, payload = call(mw, b"card 4111 1111 1111 1111")
    assert status == 413
    assert route.bodies == []
    assert b"4111" not in payload


def test_a_failed_scan_is_refused_by_default() -> None:
    route = Route()
    mw = example.WardcatMiddleware(route, guard=BrokenGuard())
    status, payload = call(mw, b"card 4111 1111 1111 1111")
    assert status == 503
    assert route.bodies == []
    assert json.loads(payload) == {"error": "PII scan unavailable; request not forwarded."}


def test_passing_on_scan_error_is_an_explicit_choice() -> None:
    route = Route()
    mw = example.WardcatMiddleware(route, guard=BrokenGuard(), on_scan_error="pass")
    status, _ = call(mw, b"card 4111 1111 1111 1111")
    assert status == 200
    assert route.bodies == [b"card 4111 1111 1111 1111"]


def test_an_unknown_on_scan_error_is_rejected() -> None:
    with pytest.raises(ValueError, match="on_scan_error"):
        example.WardcatMiddleware(Route(), guard=guard(), on_scan_error="ignore")


def test_block_mode_answers_422_without_values() -> None:
    route = Route()
    mw = example.WardcatMiddleware(route, guard=guard(), on_pii_detected="block")
    status, payload = call(
        mw, b'{"text": "card 4111 1111 1111 1111"}', content_type=b"application/json"
    )
    assert status == 422
    assert route.bodies == []
    assert b"4111" not in payload

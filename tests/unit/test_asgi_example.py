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


# ── Content type and body size ────────────────────────────────────────────────
# The content type was matched as a case-sensitive substring, so
# "Application/JSON", a form, a multipart upload or a request with no
# content-type header reached the route unscanned, against the docstring.


def call_raw(middleware, chunks: list[bytes], headers: list[tuple[bytes, bytes]]):
    """Send *chunks* as a streamed body; return (status, payload, chunks read)."""
    sent: list[dict] = []
    read = 0

    async def receive():
        nonlocal read
        read += 1
        body = chunks[read - 1]
        return {"type": "http.request", "body": body, "more_body": read < len(chunks)}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "path": "/echo", "headers": headers}
    asyncio.run(middleware(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    payload = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], payload, read


CARD_JSON = b'{"text": "card 4111 1111 1111 1111"}'


@pytest.mark.parametrize(
    "content_type",
    [b"Application/JSON", b"application/json; charset=utf-8", b"TEXT/PLAIN;charset=UTF-8"],
)
def test_the_content_type_is_matched_case_insensitively(content_type: bytes) -> None:
    route = Route()
    mw = example.WardcatMiddleware(route, guard=guard())
    status, _ = call(mw, CARD_JSON, content_type=content_type)
    assert status == 200
    assert len(route.bodies) == 1
    assert b"4111" not in route.bodies[0]


def test_a_json_suffix_type_is_scanned_as_json() -> None:
    route = Route()
    mw = example.WardcatMiddleware(route, guard=guard(), content_types=("application/",))
    call(mw, CARD_JSON, content_type=b"application/vnd.api+json")
    assert json.loads(route.bodies[0]) == {"text": "card [CREDIT_CARD]"}


def test_a_request_without_a_content_type_is_scanned_as_text() -> None:
    route = Route()
    mw = example.WardcatMiddleware(route, guard=guard())
    status, _, _ = call_raw(mw, [b"card 4111 1111 1111 1111"], headers=[])
    assert status == 200
    assert route.bodies == [b"card [CREDIT_CARD]"]


@pytest.mark.parametrize(
    "content_type",
    [b"application/x-www-form-urlencoded", b"multipart/form-data; boundary=x", b"image/png"],
)
def test_a_type_it_cannot_scan_is_refused_by_default(content_type: bytes) -> None:
    route = Route()
    mw = example.WardcatMiddleware(route, guard=guard())
    status, payload = call(mw, b"card=4111111111111111", content_type=content_type)
    assert status == 415
    assert route.bodies == []
    assert b"4111" not in payload


def test_passing_an_unscannable_type_is_an_explicit_choice() -> None:
    route = Route()
    mw = example.WardcatMiddleware(route, guard=guard(), on_unscanned="pass")
    status, _ = call(mw, b"card=4111111111111111", content_type=b"multipart/form-data")
    assert status == 200
    assert route.bodies == [b"card=4111111111111111"]


def test_an_unknown_on_unscanned_is_rejected() -> None:
    with pytest.raises(ValueError, match="on_unscanned"):
        example.WardcatMiddleware(Route(), guard=guard(), on_unscanned="ignore")


def test_an_oversized_streamed_body_is_not_read_to_the_end() -> None:
    route = Route()
    mw = example.WardcatMiddleware(route, guard=guard(), max_body_bytes=100)
    chunks = [b"x" * 60] * 50  # 3000 bytes in 60-byte parts
    status, _, read = call_raw(mw, chunks, headers=[(b"content-type", b"text/plain")])
    assert status == 413
    assert route.bodies == []
    # It stops at the part that crosses the limit, not after all fifty.
    assert read == 2

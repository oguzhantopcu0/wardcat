"""
Tests for ai_guard.integrations.fastapi.AIGuardMiddleware.

Uses httpx.AsyncClient with a minimal ASGI app to simulate real requests.
No FastAPI dependency required — middleware is ASGI-compatible.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from ai_guard import LLMGuard
from ai_guard.integrations.fastapi import AIGuardMiddleware


# ---------------------------------------------------------------------------
# Minimal ASGI app for testing
# ---------------------------------------------------------------------------

async def _echo_app(scope, receive, send):
    """ASGI app that reads the request body and echoes it back."""
    body_chunks = []
    more_body = True
    while more_body:
        message = await receive()
        body_chunks.append(message.get("body", b""))
        more_body = message.get("more_body", False)
    body = b"".join(body_chunks)

    response_body = json.dumps({
        "echo": body.decode("utf-8", errors="replace"),
        "ai_guard": scope.get("state", {}).get("ai_guard_result") is not None,
    }).encode("utf-8")

    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(response_body)).encode()),
        ],
    })
    await send({"type": "http.response.body", "body": response_body, "more_body": False})


def _make_request_scope(path: str = "/", content_type: str = "application/json") -> dict:
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [
            (b"content-type", content_type.encode()),
        ],
        "state": {},
    }


def _make_receive(body: bytes):
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    return receive


class _MockSend:
    """Collects ASGI send messages."""

    def __init__(self):
        self.messages = []

    async def __call__(self, message):
        self.messages.append(message)

    @property
    def status(self):
        for m in self.messages:
            if m.get("type") == "http.response.start":
                return m["status"]
        return None

    @property
    def body(self):
        for m in self.messages:
            if m.get("type") == "http.response.body":
                return m.get("body", b"")
        return b""


@pytest.fixture
def guard():
    g = LLMGuard(use_ner=False)
    g.configure_entity("EMAIL", enabled=True, action="warn")
    g.configure_entity("CREDIT_CARD", enabled=True, action="hash")
    return g


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestMiddlewareConstructor:
    def test_invalid_mode_raises(self, guard):
        with pytest.raises(ValueError, match="on_pii_detected"):
            AIGuardMiddleware(_echo_app, guard=guard, on_pii_detected="delete")

    def test_valid_modes_accepted(self, guard):
        for mode in ("block", "sanitize", "warn"):
            mw = AIGuardMiddleware(_echo_app, guard=guard, on_pii_detected=mode)
            assert mw.on_pii_detected == mode


# ---------------------------------------------------------------------------
# Non-HTTP scope passthrough
# ---------------------------------------------------------------------------

class TestNonHttpScope:
    def test_websocket_scope_passed_through(self, guard):
        mw = AIGuardMiddleware(_echo_app, guard=guard)
        calls = []

        async def fake_app(scope, receive, send):
            calls.append(scope["type"])

        mw2 = AIGuardMiddleware(fake_app, guard=guard)

        async def _run():
            scope = {"type": "websocket", "path": "/ws"}
            await mw2(scope, None, None)

        asyncio.run(_run())
        assert "websocket" in calls


# ---------------------------------------------------------------------------
# Path prefix filtering
# ---------------------------------------------------------------------------

class TestPathPrefixFilter:
    def test_non_matching_path_not_scanned(self, guard):
        mw = AIGuardMiddleware(_echo_app, guard=guard, scan_path_prefix="/api/")
        send = _MockSend()
        body = b'{"text": "user@example.com"}'
        scope = _make_request_scope(path="/static/file.txt")

        async def _run():
            await mw(scope, _make_receive(body), send)

        asyncio.run(_run())
        # Echo app ran — state should not have ai_guard_result (middleware skipped)
        # (state was not injected by middleware since path didn't match)
        response = json.loads(send.body)
        assert response["ai_guard"] is False

    def test_matching_path_is_scanned(self, guard):
        mw = AIGuardMiddleware(_echo_app, guard=guard, scan_path_prefix="/api/")
        send = _MockSend()
        body = b'{"text": "user@example.com"}'
        scope = _make_request_scope(path="/api/chat")

        async def _run():
            await mw(scope, _make_receive(body), send)

        asyncio.run(_run())
        response = json.loads(send.body)
        assert response["ai_guard"] is True


# ---------------------------------------------------------------------------
# Content-type filtering
# ---------------------------------------------------------------------------

class TestContentTypeFilter:
    def test_binary_content_type_skipped(self, guard):
        mw = AIGuardMiddleware(_echo_app, guard=guard)
        send = _MockSend()
        body = b"binary content"
        scope = _make_request_scope(content_type="application/octet-stream")

        async def _run():
            await mw(scope, _make_receive(body), send)

        asyncio.run(_run())
        response = json.loads(send.body)
        assert response["ai_guard"] is False  # middleware skipped

    def test_json_content_type_scanned(self, guard):
        mw = AIGuardMiddleware(_echo_app, guard=guard)
        send = _MockSend()
        body = b'{"email": "user@example.com"}'
        scope = _make_request_scope(content_type="application/json")

        async def _run():
            await mw(scope, _make_receive(body), send)

        asyncio.run(_run())
        response = json.loads(send.body)
        assert response["ai_guard"] is True


# ---------------------------------------------------------------------------
# Mode: warn
# ---------------------------------------------------------------------------

class TestWarnMode:
    def test_warn_mode_passes_through_original_body(self, guard):
        mw = AIGuardMiddleware(_echo_app, guard=guard, on_pii_detected="warn")
        send = _MockSend()
        original = "email: user@example.com"
        scope = _make_request_scope(content_type="text/plain")

        async def _run():
            await mw(scope, _make_receive(original.encode()), send)

        asyncio.run(_run())
        response = json.loads(send.body)
        assert "user@example.com" in response["echo"]

    def test_warn_mode_clean_text_passes_through(self, guard):
        mw = AIGuardMiddleware(_echo_app, guard=guard, on_pii_detected="warn")
        send = _MockSend()
        body = b"clean text, no PII here"
        scope = _make_request_scope(content_type="text/plain")

        async def _run():
            await mw(scope, _make_receive(body), send)

        asyncio.run(_run())
        assert send.status == 200


# ---------------------------------------------------------------------------
# Mode: block
# ---------------------------------------------------------------------------

class TestBlockMode:
    def test_block_mode_returns_422_on_pii(self, guard):
        mw = AIGuardMiddleware(_echo_app, guard=guard, on_pii_detected="block")
        send = _MockSend()
        body = b"email: user@example.com"
        scope = _make_request_scope(content_type="text/plain")

        async def _run():
            await mw(scope, _make_receive(body), send)

        asyncio.run(_run())
        assert send.status == 422
        resp = json.loads(send.body)
        assert "error" in resp
        assert "violations" in resp
        # raw PII must not appear in error response
        assert "user@example.com" not in send.body.decode()

    def test_block_mode_clean_text_passes(self, guard):
        mw = AIGuardMiddleware(_echo_app, guard=guard, on_pii_detected="block")
        send = _MockSend()
        body = b"clean text, no PII"
        scope = _make_request_scope(content_type="text/plain")

        async def _run():
            await mw(scope, _make_receive(body), send)

        asyncio.run(_run())
        assert send.status == 200


# ---------------------------------------------------------------------------
# Mode: sanitize
# ---------------------------------------------------------------------------

class TestSanitizeMode:
    def test_sanitize_replaces_pii_in_forwarded_body(self):
        guard = LLMGuard(use_ner=False)
        guard.configure_entity("EMAIL", enabled=True, action="redact")
        mw = AIGuardMiddleware(_echo_app, guard=guard, on_pii_detected="sanitize")
        send = _MockSend()
        body = b"email: user@example.com"
        scope = _make_request_scope(content_type="text/plain")

        async def _run():
            await mw(scope, _make_receive(body), send)

        asyncio.run(_run())
        assert send.status == 200
        response = json.loads(send.body)
        assert "user@example.com" not in response["echo"]
        assert "[EMAIL]" in response["echo"]

    def test_sanitize_clean_text_unchanged(self, guard):
        mw = AIGuardMiddleware(_echo_app, guard=guard, on_pii_detected="sanitize")
        send = _MockSend()
        body = b"clean text"
        scope = _make_request_scope(content_type="text/plain")

        async def _run():
            await mw(scope, _make_receive(body), send)

        asyncio.run(_run())
        response = json.loads(send.body)
        assert response["echo"] == "clean text"


# ---------------------------------------------------------------------------
# Body size limit
# ---------------------------------------------------------------------------

class TestBodySizeLimit:
    def test_oversized_body_passed_through_unscanned(self, guard):
        mw = AIGuardMiddleware(_echo_app, guard=guard, on_pii_detected="block",
                               max_body_bytes=10)
        send = _MockSend()
        body = b"user@example.com and more text"
        scope = _make_request_scope(content_type="text/plain")

        async def _run():
            await mw(scope, _make_receive(body), send)

        asyncio.run(_run())
        # Body too large — should not be blocked (scan skipped)
        assert send.status == 200

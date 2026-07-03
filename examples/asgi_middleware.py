"""
Example: a self-contained ASGI middleware built on top of wardcat.

wardcat is a *library* — it deliberately ships no web-framework code. If you
want to scan request bodies automatically, copy this middleware into your app.
It is pure ASGI (works with FastAPI, Starlette, Quart, …) and only depends on
the public wardcat API.

Modes (on_pii_detected):
    "block"    → reject requests containing PII with HTTP 422
    "sanitize" → replace PII in the body before it reaches the route
    "warn"     → pass through unchanged; result on scope["state"]["wardcat_result"]

Run:
    pip install "wardcat" fastapi uvicorn
    uvicorn examples.asgi_middleware:app --reload
    curl -s -X POST localhost:8000/echo -H 'content-type: application/json' \
         -d '{"text":"my card is 4111 1111 1111 1111"}'
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from wardcat import Wardcat
from wardcat.core.models import ScanResult

logger = logging.getLogger(__name__)

_DEFAULT_CONTENT_TYPES: tuple[str, ...] = ("application/json", "text/plain", "text/")


class WardcatMiddleware:
    """ASGI middleware that scans HTTP request bodies for PII with an Wardcat."""

    def __init__(
        self,
        app,
        *,
        guard: Wardcat,
        on_pii_detected: str = "sanitize",
        scan_path_prefix: str | None = None,
        content_types: Sequence[str] = _DEFAULT_CONTENT_TYPES,
        max_body_bytes: int = 1_000_000,
        json_field_scan: bool = True,
    ) -> None:
        if on_pii_detected not in ("block", "sanitize", "warn"):
            raise ValueError(
                f"Invalid on_pii_detected={on_pii_detected!r}. "
                "Valid values: 'block', 'sanitize', 'warn'."
            )
        self.app = app
        self.guard = guard
        self.on_pii_detected = on_pii_detected
        self.scan_path_prefix = scan_path_prefix
        self.content_types = tuple(content_types)
        self.max_body_bytes = max_body_bytes
        self.json_field_scan = json_field_scan

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")
        if self.scan_path_prefix and not path.startswith(self.scan_path_prefix):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_type = headers.get(b"content-type", b"").decode("latin-1", errors="replace")
        if not any(ct in content_type for ct in self.content_types):
            await self.app(scope, receive, send)
            return

        body_chunks: list[bytes] = []
        more_body = True
        while more_body:
            message = await receive()
            body_chunks.append(message.get("body", b""))
            more_body = message.get("more_body", False)
        body = b"".join(body_chunks)

        if len(body) > self.max_body_bytes:
            await self._forward(scope, send, body)
            return

        text = body.decode("utf-8", errors="replace")
        is_json = "application/json" in content_type

        try:
            if is_json and self.json_field_scan:
                sanitized_body, result = await self._scan_json_body(text)
            else:
                result = await self.guard.scan_async(text)
                sanitized_body = result.sanitized_text.encode("utf-8")
        except Exception as exc:
            logger.warning("wardcat: scan failed (%s) — passing request through.", exc)
            await self._forward(scope, send, body)
            return

        scope.setdefault("state", {})["wardcat_result"] = result

        if result.is_clean or self.on_pii_detected == "warn":
            await self._forward(scope, send, body)
            return

        if self.on_pii_detected == "block":
            await self._send_422(send, result.redacted())
            return

        await self._forward(scope, send, sanitized_body)

    async def _scan_json_body(self, text: str):
        """Parse JSON, scan each string value, reconstruct — keys/structure preserved."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            result = await self.guard.scan_async(text)
            return result.sanitized_text.encode("utf-8"), result

        all_violations: list = []

        async def _process(node: Any) -> Any:
            if isinstance(node, str):
                r = await self.guard.scan_async(node)
                all_violations.extend(r.violations)
                return r.sanitized_text
            if isinstance(node, dict):
                return {k: await _process(v) for k, v in node.items()}
            if isinstance(node, list):
                return [await _process(item) for item in node]
            return node

        sanitized_json = json.dumps(await _process(data), ensure_ascii=False)
        result = ScanResult(
            original_text=text, sanitized_text=sanitized_json, violations=all_violations
        )
        return sanitized_json.encode("utf-8"), result

    async def _forward(self, scope, send, body: bytes) -> None:
        sent = False

        async def patched_receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        await self.app(scope, patched_receive, send)

    @staticmethod
    async def _send_422(send, redacted: dict) -> None:
        body = json.dumps(
            {
                "error": "Request blocked: PII detected.",
                "violations": redacted.get("violations", []),
            }
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 422,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})


# ── Demo app ────────────────────────────────────────────────────────────────

try:
    from fastapi import FastAPI, Request

    guard = (
        Wardcat(use_ner=False, salt="example-salt")
        .add_entity("EMAIL", action="warn")
        .add_entity("CREDIT_CARD", action="hash")
    )

    app = FastAPI()
    app.add_middleware(WardcatMiddleware, guard=guard, on_pii_detected="sanitize")

    @app.post("/echo")
    async def echo(request: Request) -> dict:
        body = await request.body()
        result = request.scope.get("state", {}).get("wardcat_result")
        return {
            "received": body.decode("utf-8", errors="replace"),
            "had_pii": bool(result and not result.is_clean),
        }

except ImportError:
    # FastAPI is only needed to run the demo app, not the middleware itself.
    pass

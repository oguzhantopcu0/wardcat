"""
FastAPI / Starlette ASGI middleware for PII detection.

Usage::

    from fastapi import FastAPI
    from ai_guard import LLMGuard
    from ai_guard.integrations.fastapi import AIGuardMiddleware

    app = FastAPI()
    guard = LLMGuard(use_ner=False, salt="my-secret-salt")

    app.add_middleware(
        AIGuardMiddleware,
        guard=guard,
        on_pii_detected="block",          # "block" | "sanitize" | "warn"
        scan_path_prefix="/api/",          # only scan these routes (optional)
        content_types=("application/json", "text/plain"),
        json_field_scan=True,              # scan JSON string values individually
    )

Request state
-------------
After the middleware runs, ``request.state.ai_guard_result`` holds the
:class:`~ai_guard.core.models.ScanResult` for the request.  For JSON bodies
with ``json_field_scan=True``, it is a *synthetic* result whose
``violations`` list aggregates all per-field violations.

Modes
-----
``"block"``
    Return HTTP 422 immediately if PII is detected.  The response body
    contains ``{"error": "...", "violations": [...]}`` (no raw PII).

``"sanitize"``
    Forward the sanitized body to the endpoint.  For JSON bodies with
    ``json_field_scan=True``, only the affected string *values* are
    replaced; JSON structure and keys are preserved.

``"warn"``
    Log the violation and forward the original body unchanged.

JSON field scanning (``json_field_scan=True``)
----------------------------------------------
When the request ``Content-Type`` is ``application/json`` and
``json_field_scan=True``, the middleware parses the JSON, scans each
string value independently with the regex guard, and reconstructs the
JSON with sanitized values.  JSON structure (keys, nesting, arrays) is
preserved exactly.

.. note::
    For LLM-enabled guards, ``json_field_scan=True`` may be slow because
    each string field would trigger a separate LLM call.  Set
    ``json_field_scan=False`` to scan the raw JSON text instead.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# Default MIME types to scan. Binary content types are skipped.
_DEFAULT_CONTENT_TYPES: tuple[str, ...] = ("application/json", "text/plain", "text/")


class AIGuardMiddleware:
    """ASGI middleware that scans HTTP request bodies for PII.

    Compatible with FastAPI and any Starlette-based application.

    :param app:              The wrapped ASGI application.
    :param guard:            Configured :class:`~ai_guard.LLMGuard` instance.
    :param on_pii_detected:  Action when PII is found — ``"block"``,
                             ``"sanitize"``, or ``"warn"`` (default: ``"sanitize"``).
    :param scan_path_prefix: If set, only paths starting with this prefix are
                             scanned.  E.g. ``"/api/"`` skips static files.
    :param content_types:    MIME type prefixes to scan.  Requests whose
                             ``Content-Type`` does not match any prefix are
                             passed through untouched.
    :param max_body_bytes:   Maximum request body size to scan in bytes.
                             Larger bodies are passed through unchanged
                             (default: 1 MB).
    :param json_field_scan:  When ``True`` and ``Content-Type`` is
                             ``application/json``, scan each JSON string value
                             individually so keys and structure are never
                             modified (default: ``True``).
    """

    def __init__(
        self,
        app,
        *,
        guard,
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

        # Buffer the request body
        body_chunks: list[bytes] = []
        more_body = True
        while more_body:
            message = await receive()
            body_chunks.append(message.get("body", b""))
            more_body = message.get("more_body", False)

        body = b"".join(body_chunks)

        if len(body) > self.max_body_bytes:
            logger.debug(
                "ai-guard: request body (%d bytes) exceeds max_body_bytes (%d) — skipped.",
                len(body), self.max_body_bytes,
            )
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
            logger.warning("ai-guard: scan failed (%s) — passing request through.", exc)
            await self._forward(scope, send, body)
            return

        # Attach result to scope state so endpoints can access it via request.state
        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["ai_guard_result"] = result

        if result.is_clean or self.on_pii_detected == "warn":
            if not result.is_clean:
                logger.warning(
                    "ai-guard: PII detected in request to %s — %d violation(s). "
                    "(mode=warn, passing through)",
                    path, len(result.violations),
                )
            await self._forward(scope, send, body)
            return

        if self.on_pii_detected == "block":
            logger.info(
                "ai-guard: blocked request to %s — %d PII violation(s).",
                path, len(result.violations),
            )
            await self._send_422(send, result.redacted())
            return

        # sanitize mode: forward sanitized body downstream
        logger.info(
            "ai-guard: sanitized %d violation(s) in request to %s.",
            len(result.violations), path,
        )
        await self._forward(scope, send, sanitized_body)

    # ------------------------------------------------------------------
    # JSON field-level scanning
    # ------------------------------------------------------------------

    async def _scan_json_body(self, text: str):
        """Parse JSON, scan each string value, reconstruct JSON.

        Returns ``(sanitized_body_bytes, synthetic_ScanResult)``.
        The ScanResult aggregates all per-field violations; its
        ``original_text`` and ``sanitized_text`` are set to the full JSON
        strings (before and after sanitization).
        """
        from ai_guard.core.models import ScanResult

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Not parseable as JSON — fall back to full-text scan
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

        sanitized_data = await _process(data)
        sanitized_json = json.dumps(sanitized_data, ensure_ascii=False)
        sanitized_body = sanitized_json.encode("utf-8")

        result = ScanResult(
            original_text=text,
            sanitized_text=sanitized_json,
            violations=all_violations,
        )
        return sanitized_body, result

    # ------------------------------------------------------------------

    async def _forward(self, scope, send, body: bytes) -> None:
        """Forward the request to the downstream app with the given body."""
        body_sent = False

        async def patched_receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        await self.app(scope, patched_receive, send)

    @staticmethod
    async def _send_422(send, redacted: dict) -> None:
        """Send an HTTP 422 response with violation metadata (no raw PII)."""
        body = json.dumps({
            "error": "Request blocked: PII detected.",
            "violations": redacted.get("violations", []),
        }).encode("utf-8")

        await send({
            "type": "http.response.start",
            "status": 422,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": body,
            "more_body": False,
        })

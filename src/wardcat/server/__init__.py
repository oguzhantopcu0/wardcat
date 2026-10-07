"""The HTTP service behind ``wardcat serve``. Not public API.

The policy is fixed when the service starts; nothing over HTTP can change it.
(wardcat-cli 0.5 exposed ``POST /layers`` and ``POST /filters``: an
unauthenticated local process could switch masking off for every client, or
point the LLM layer at its own server and receive every text scanned after.)

Endpoints:

========================  ======  ====================================================
``GET  /healthz``         open    the process is up
``GET  /readyz``          open    the guard is built and not degraded under strict
``GET  /info``            key     enabled entity types with their actions, and layers
``POST /scan``            key     ``{"text": ...}`` → the ``redacted()`` dict
``POST /is-sensitive``    key     ``{"text": ...}`` → ``{"sensitive": bool}``
``GET  /metrics``         key     request counters, Prometheus text format
========================  ======  ====================================================

"key" means the API key is required when one is configured. Without one the
service only runs on a loopback address, and then checks the ``Host`` header
against loopback names, so a web page cannot reach it through DNS rebinding.

Every error body is one of a fixed set of codes, never an exception message and
never request text; the access log records method, path, status and duration.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from collections import Counter
from collections.abc import Awaitable, Callable, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from wardcat.server._hosts import host_is_loopback

if TYPE_CHECKING:  # pragma: no cover
    from wardcat import Wardcat

logger = logging.getLogger("wardcat.server")

# Room for the JSON envelope around a text at the guard's own size limit.
_ENVELOPE_BYTES = 64 * 1024
_OPEN_PATHS = frozenset({"/healthz", "/readyz"})


class _Refused(Exception):
    def __init__(self, status: int, code: str) -> None:
        self.status, self.code = status, code


@dataclass
class ServerConfig:
    api_key: str | None = None
    cors_origins: Sequence[str] = ()
    max_concurrency: int = 8
    request_timeout: float = 30.0
    stats: Counter[str] = field(default_factory=Counter)


def _error(status: int, code: str) -> JSONResponse:
    return JSONResponse({"error": code}, status_code=status)


def create_app(guard: Wardcat, config: ServerConfig | None = None) -> Starlette:
    """The ASGI app serving *guard*, whose policy cannot change while it runs."""
    cfg = config or ServerConfig()
    body_limit = guard.max_text_bytes + _ENVELOPE_BYTES
    slots = asyncio.Semaphore(max(1, cfg.max_concurrency))
    has_llm = guard._llm_detector is not None

    async def read_text(request: Request) -> str:
        content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type != "application/json":
            # Requiring JSON forces a CORS preflight on any cross-origin request,
            # so a page cannot fire "simple" POSTs at a loopback service.
            raise _Refused(415, "unsupported_media_type")
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > body_limit:
            raise _Refused(413, "too_large")
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > body_limit:
                raise _Refused(413, "too_large")
            chunks.append(chunk)
        try:
            payload = json.loads(b"".join(chunks).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise _Refused(400, "bad_request") from None
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str):
            raise _Refused(400, "bad_request")
        return text

    async def healthz(request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def readyz(request: Request) -> Response:
        if guard._config.get("strict") and guard._engine.build_warnings:
            return _error(503, "degraded")
        return JSONResponse({"status": "ready"})

    async def info(request: Request) -> Response:
        layers = ["regex"]
        if guard._config.get("use_ner") and any(
            type(d).__name__ == "NERDetector" for d in guard._detectors
        ):
            layers.append("ner")
        if has_llm:
            layers.append("llm")
        return JSONResponse(
            {
                "entities": guard.entity_policy(),
                "layers": layers,
                "warnings": list(guard._engine.build_warnings),
            }
        )

    async def scan(request: Request) -> Response:
        from wardcat.exceptions import DegradedScanError

        text = await read_text(request)
        try:
            result = await guard.scan_async(text)
        except DegradedScanError:
            raise _Refused(503, "degraded") from None
        except ValueError:
            raise _Refused(413, "too_large") from None
        return JSONResponse(result.redacted())

    async def is_sensitive(request: Request) -> Response:
        import httpx

        from wardcat.llm.circuit import CircuitOpen

        if not has_llm:
            raise _Refused(409, "llm_not_configured")
        text = await read_text(request)
        try:
            verdict = await guard.is_sensitive_async(text)
        except (ConnectionError, TimeoutError, httpx.HTTPError, CircuitOpen):
            raise _Refused(503, "llm_unavailable") from None
        except ValueError:
            raise _Refused(413, "too_large") from None
        return JSONResponse({"sensitive": bool(verdict)})

    async def metrics(request: Request) -> Response:
        lines = ["# TYPE wardcat_requests_total counter"]
        for key, value in sorted(cfg.stats.items()):
            path, status = key.split(" ", 1)
            lines.append(f'wardcat_requests_total{{path="{path}",status="{status}"}} {value}')
        return PlainTextResponse("\n".join(lines) + "\n")

    def guarded(
        handler: Callable[[Request], Awaitable[Response]],
    ) -> Callable[[Request], Awaitable[Response]]:
        async def endpoint(request: Request) -> Response:
            if slots.locked():
                return _error(503, "busy")
            async with slots:
                try:
                    return await asyncio.wait_for(handler(request), timeout=cfg.request_timeout)
                except _Refused as exc:
                    return _error(exc.status, exc.code)
                except TimeoutError:
                    return _error(503, "timeout")
                except Exception as exc:
                    # The type only: an exception message can quote its input.
                    logger.error("server: unhandled %s on %s", type(exc).__name__, request.url.path)
                    return _error(500, "internal_error")

        return endpoint

    routes = [
        Route("/healthz", healthz, methods=["GET"]),
        Route("/readyz", readyz, methods=["GET"]),
        Route("/info", guarded(info), methods=["GET"]),
        Route("/scan", guarded(scan), methods=["POST"]),
        Route("/is-sensitive", guarded(is_sensitive), methods=["POST"]),
        Route("/metrics", metrics, methods=["GET"]),
    ]
    app = Starlette(routes=routes, debug=False)
    app.add_middleware(_GateMiddleware, cfg=cfg)
    if cfg.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cfg.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["authorization", "x-api-key", "content-type"],
            allow_credentials=False,
        )
    return app


class _GateMiddleware:
    """Host, origin and key checks, plus the access log and counters, in one place."""

    def __init__(self, app: Any, cfg: ServerConfig) -> None:
        self.app, self.cfg = app, cfg

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        status_holder = {"status": 0}

        async def send_with_status(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
            await send(message)

        refusal = self._refusal(scope)
        if refusal is not None:
            await refusal(scope, receive, send_with_status)
        else:
            await self.app(scope, receive, send_with_status)
        path = scope.get("path", "")
        status = status_holder["status"]
        self.cfg.stats[f"{path} {status}"] += 1
        logger.info(
            "%s %s %d %.1fms",
            scope.get("method", ""),
            path,
            status,
            (time.perf_counter() - started) * 1000,
        )

    def _refusal(self, scope: dict) -> Response | None:
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        path = scope.get("path", "")
        if not self.cfg.api_key:
            # Keyless only ever runs on loopback (the CLI refuses otherwise); the
            # Host check closes DNS rebinding from a web page.
            if not host_is_loopback(headers.get("host", "")):
                return _error(400, "bad_host")
        if "origin" in headers and not self.cfg.cors_origins:
            return _error(403, "cross_origin")
        if path in _OPEN_PATHS:
            return None
        if scope.get("method") == "OPTIONS" and self.cfg.cors_origins:
            return None  # a CORS preflight; CORSMiddleware answers it
        if self.cfg.api_key and not _key_matches(headers, self.cfg.api_key):
            return _error(401, "unauthorized")
        return None


def _key_matches(headers: dict[str, str], key: str) -> bool:
    expected = key.encode()
    bearer = headers.get("authorization", "")
    candidates = []
    if bearer[:7].lower() == "bearer ":
        candidates.append(bearer[7:].strip())
    if "x-api-key" in headers:
        candidates.append(headers["x-api-key"].strip())
    # Compare every candidate, so the time taken does not say which header matched.
    results = [hmac.compare_digest(c.encode(), expected) for c in candidates]
    return any(results)

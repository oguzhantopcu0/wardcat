"""``wardcat serve``: what the service refuses, and that it never leaks a value.

wardcat-cli 0.5 served the same guard with a mutable policy (POST /layers,
POST /filters), no key by default and any bind address. These tests pin the
replacement's contract: the policy cannot change over HTTP, a key is required
off loopback, keyless service answers only loopback Host headers, and every
error body is a fixed code.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

pytest.importorskip("starlette")

from starlette.testclient import TestClient  # noqa: E402

from wardcat import Wardcat  # noqa: E402
from wardcat.cli import EXIT_CONFIG, main  # noqa: E402
from wardcat.server import ServerConfig, create_app  # noqa: E402
from wardcat.server._hosts import address_is_loopback, host_is_loopback  # noqa: E402

CARD = "4111 1111 1111 1111"
KEY = "k" * 32
TEXT = {"text": f"mail ali@example.com card {CARD}"}


def guard() -> Wardcat:
    return Wardcat(salt="s").add_entities(["EMAIL", "CREDIT_CARD"], action="redact")


def client(*, key: str | None = None, host: str = "127.0.0.1:8787", **cfg) -> TestClient:
    app = create_app(guard(), ServerConfig(api_key=key, **cfg))
    return TestClient(app, base_url=f"http://{host}")


def auth(kind: str = "bearer") -> dict[str, str]:
    return {"authorization": f"Bearer {KEY}"} if kind == "bearer" else {"x-api-key": KEY}


class TestScan:
    def test_scan_returns_the_redacted_dict(self) -> None:
        r = client().post("/scan", json=TEXT)
        assert r.status_code == 200
        body = r.json()
        assert body["sanitized_text"] == "mail [EMAIL] card [CREDIT_CARD]"
        assert body["is_clean"] is False
        assert CARD not in r.text and "ali@example.com" not in r.text

    def test_health_and_ready(self) -> None:
        c = client()
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.get("/readyz").json() == {"status": "ready"}

    def test_info_lists_the_policy(self) -> None:
        body = client().get("/info").json()
        assert body["entities"] == {"CREDIT_CARD": "redact", "EMAIL": "redact"}
        assert body["layers"] == ["regex"]

    def test_is_sensitive_without_an_llm_is_409(self) -> None:
        r = client().post("/is-sensitive", json={"text": "x"})
        assert (r.status_code, r.json()) == (409, {"error": "llm_not_configured"})


class TestThePolicyCannotChange:
    @pytest.mark.parametrize("path", ["/layers", "/filters"])
    def test_wardcat_cli_mutation_endpoints_do_not_exist(self, path: str) -> None:
        r = client().post(path, json={"layer": "llm", "base_url": "https://attacker.invalid/v1"})
        assert r.status_code in (404, 405)

    def test_no_request_changes_the_guard(self) -> None:
        g = guard()
        before = (g.entity_policy(), g._config.get("llm_detector"))
        c = TestClient(create_app(g), base_url="http://127.0.0.1")
        for method, path in [
            ("get", "/info"),
            ("get", "/metrics"),
            ("post", "/scan"),
            ("post", "/is-sensitive"),
            ("post", "/layers"),
            ("put", "/info"),
        ]:
            c.request(
                method.upper(), path, json={"text": "a@b.io", "layer": "llm", "enabled": False}
            )
        assert (g.entity_policy(), g._config.get("llm_detector")) == before


class TestAuth:
    def test_a_key_is_required_everywhere_but_health(self) -> None:
        c = client(key=KEY)
        assert c.post("/scan", json=TEXT).status_code == 401
        assert c.get("/info").status_code == 401
        assert c.get("/metrics").status_code == 401
        assert c.get("/healthz").status_code == 200
        assert c.get("/readyz").status_code == 200

    @pytest.mark.parametrize("kind", ["bearer", "x-api-key"])
    def test_both_header_forms_work(self, kind: str) -> None:
        assert client(key=KEY).post("/scan", json=TEXT, headers=auth(kind)).status_code == 200

    def test_a_wrong_key_is_401(self) -> None:
        r = client(key=KEY).post("/scan", json=TEXT, headers={"x-api-key": "nope"})
        assert (r.status_code, r.json()) == (401, {"error": "unauthorized"})

    def test_with_a_key_any_host_header_is_fine(self) -> None:
        r = client(key=KEY, host="wardcat.internal:8787").post("/scan", json=TEXT, headers=auth())
        assert r.status_code == 200


class TestKeylessIsLoopbackOnly:
    @pytest.mark.parametrize(
        "host", ["127.0.0.1:8787", "localhost:8787", "[::1]:8787", "[::1]", "127.0.0.2:9000"]
    )
    def test_loopback_host_headers_are_served(self, host: str) -> None:
        assert client(host=host).get("/info").status_code == 200

    @pytest.mark.parametrize("host", ["evil.example:8787", "evil.example", "10.0.0.5:8787"])
    def test_other_host_headers_are_refused(self, host: str) -> None:
        r = client(host=host).get("/info")
        assert (r.status_code, r.json()) == (400, {"error": "bad_host"})

    def test_a_page_cannot_post_text_plain(self) -> None:
        r = client().post("/scan", content=json.dumps(TEXT), headers={"content-type": "text/plain"})
        assert (r.status_code, r.json()) == (415, {"error": "unsupported_media_type"})

    def test_a_cross_origin_request_is_refused_without_cors(self) -> None:
        r = client().post("/scan", json=TEXT, headers={"origin": "https://evil.example"})
        assert (r.status_code, r.json()) == (403, {"error": "cross_origin"})

    def test_cors_lets_a_named_origin_through(self) -> None:
        c = client(key=KEY, cors_origins=("https://app.example",))
        pre = c.options(
            "/scan",
            headers={"origin": "https://app.example", "access-control-request-method": "POST"},
        )
        assert pre.status_code == 200
        assert pre.headers["access-control-allow-origin"] == "https://app.example"
        r = c.post("/scan", json=TEXT, headers={**auth(), "origin": "https://app.example"})
        assert r.status_code == 200


class TestLimits:
    def test_a_body_over_the_limit_is_413_without_scanning(self, monkeypatch) -> None:
        g = Wardcat(config_path=None, salt="s").add_entity("EMAIL", "redact")
        g._config["max_text_bytes"] = 1000
        calls = []
        monkeypatch.setattr(g, "scan_async", lambda text: calls.append(text))
        c = TestClient(create_app(g), base_url="http://127.0.0.1")
        r = c.post("/scan", json={"text": "x" * 200_000})
        assert (r.status_code, r.json()) == (413, {"error": "too_large"}) and calls == []

    @pytest.mark.parametrize("body", [b"not json", b"[1, 2]", b'{"text": 5}', b"{}"])
    def test_a_malformed_body_is_400_and_not_echoed(self, body: bytes) -> None:
        r = client().post("/scan", content=body, headers={"content-type": "application/json"})
        assert (r.status_code, r.json()) == (400, {"error": "bad_request"})

    def test_a_slow_request_times_out(self, monkeypatch) -> None:
        g = guard()

        async def slow(text):
            await asyncio.sleep(5)

        monkeypatch.setattr(g, "scan_async", slow)
        c = TestClient(
            create_app(g, ServerConfig(request_timeout=0.1)), base_url="http://127.0.0.1"
        )
        r = c.post("/scan", json=TEXT)
        assert (r.status_code, r.json()) == (503, {"error": "timeout"})

    def test_busy_beyond_the_concurrency_limit(self, monkeypatch) -> None:
        import httpx

        g = guard()
        entered, release = asyncio.Event(), asyncio.Event()

        async def held(text):
            entered.set()
            await release.wait()
            return guard().scan(text)

        monkeypatch.setattr(g, "scan_async", held)
        app = create_app(g, ServerConfig(max_concurrency=1))

        async def scenario():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as c:
                first = asyncio.create_task(c.post("/scan", json=TEXT))
                await entered.wait()  # the only slot is taken
                second = await c.post("/scan", json=TEXT)
                release.set()
                return (await first).status_code, second

        first_status, second = asyncio.run(scenario())
        assert first_status == 200
        assert (second.status_code, second.json()) == (503, {"error": "busy"})

    def test_degraded_under_strict_is_503(self) -> None:
        g = guard().with_ner(spacy_model="xx_no_such_model", auto_download=False)
        g.add_entity("PERSON", "redact")
        g._config["strict"] = True
        g._engine._strict = True
        c = TestClient(create_app(g), base_url="http://127.0.0.1")
        assert c.get("/readyz").json() == {"error": "degraded"}
        assert c.post("/scan", json=TEXT).json() == {"error": "degraded"}


class TestNoValueReachesALog:
    def test_access_log_has_no_text_or_query(self, caplog) -> None:
        with caplog.at_level(logging.DEBUG, logger="wardcat"):
            client().post("/scan?q=ali@example.com", json=TEXT)
        logged = "\n".join(r.getMessage() for r in caplog.records)
        assert "POST /scan 200" in logged
        assert CARD not in logged and "ali@example.com" not in logged

    def test_metrics_count_requests(self) -> None:
        c = client(key=KEY)
        c.post("/scan", json=TEXT, headers=auth())
        c.post("/scan", json=TEXT)
        text = c.get("/metrics", headers=auth()).text
        assert 'wardcat_requests_total{path="/scan",status="200"} 1' in text
        assert 'wardcat_requests_total{path="/scan",status="401"} 1' in text


class TestHosts:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("127.0.0.1", True),
            ("127.9.9.9", True),
            ("::1", True),
            ("[::1]", True),
            ("localhost", True),
            ("::ffff:127.0.0.1", True),
            ("0.0.0.0", False),
            ("::", False),
            ("10.0.0.1", False),
            ("myhost", False),
            ("", False),
        ],
    )
    def test_bind_addresses(self, value: str, expected: bool) -> None:
        assert address_is_loopback(value) is expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("127.0.0.1:8787", True),
            ("[::1]:8787", True),
            ("[::1]", True),
            ("LOCALHOST:1", True),
            ("[::1]x", False),
            ("[::1", False),
            ("evil.example:8787", False),
            ("", False),
        ],
    )
    def test_host_headers(self, value: str, expected: bool) -> None:
        assert host_is_loopback(value) is expected


class TestServeCommand:
    def _run(self, monkeypatch, *argv) -> tuple[int, list]:
        import uvicorn

        calls: list = []
        monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append(kw))
        return main(["serve", "--entity", "EMAIL", *argv]), calls

    def test_loopback_without_a_key_runs(self, monkeypatch) -> None:
        monkeypatch.delenv("WARDCAT_API_KEY", raising=False)
        code, calls = self._run(monkeypatch)
        assert code == 0 and calls[0]["host"] == "127.0.0.1" and calls[0]["access_log"] is False

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "10.0.0.5", "myhost"])
    def test_off_loopback_without_a_key_is_refused(self, monkeypatch, capsys, host) -> None:
        monkeypatch.delenv("WARDCAT_API_KEY", raising=False)
        code, calls = self._run(monkeypatch, "--host", host)
        assert code == EXIT_CONFIG and calls == []
        assert "without an API key" in capsys.readouterr().err

    def test_off_loopback_with_a_key_runs(self, monkeypatch) -> None:
        monkeypatch.setenv("WARDCAT_API_KEY", KEY)
        code, calls = self._run(monkeypatch, "--host", "0.0.0.0")
        assert code == 0 and calls[0]["host"] == "0.0.0.0"

    def test_a_short_key_is_refused(self, monkeypatch) -> None:
        monkeypatch.setenv("WARDCAT_API_KEY", "short")
        assert self._run(monkeypatch)[0] == EXIT_CONFIG

    def test_the_old_token_variable_is_not_silently_ignored(self, monkeypatch, capsys) -> None:
        monkeypatch.delenv("WARDCAT_API_KEY", raising=False)
        monkeypatch.setenv("WARDCAT_AUTH_TOKEN", KEY)
        code, calls = self._run(monkeypatch)
        assert (
            code == EXIT_CONFIG and calls == [] and "WARDCAT_AUTH_TOKEN" in capsys.readouterr().err
        )

    def test_cors_star_is_refused(self, monkeypatch) -> None:
        monkeypatch.delenv("WARDCAT_API_KEY", raising=False)
        assert self._run(monkeypatch, "--cors-origin", "*")[0] == EXIT_CONFIG

    def test_without_the_extra_it_says_how_to_install(self, monkeypatch, capsys) -> None:
        import builtins

        real = builtins.__import__

        def no_uvicorn(name, *a, **kw):
            if name.split(".")[0] in ("uvicorn", "starlette"):
                raise ImportError(name)
            return real(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", no_uvicorn)
        assert main(["serve", "--entity", "EMAIL"]) == EXIT_CONFIG
        assert "pip install 'wardcat[serve]'" in capsys.readouterr().err

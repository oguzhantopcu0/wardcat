"""
Production hardening tests.

Scope:
- ReDoS (Regex Denial-of-Service) protection
- Large input handling
- Thread-safety (concurrent scan)
- scan_batch error isolation
- Environment variable configuration
- Config validation
- Entity type warning
- Salt warning
- Encoding error handling (CLI)
- __version__ export
"""

from __future__ import annotations

import logging
import os
import threading
import time
from unittest.mock import patch

import pytest

from tests.conftest import make_legacy_guard
from wardcat import Wardcat, __version__
from wardcat.config.loader import load_config, validate_config
from wardcat.core.models import KNOWN_ENTITY_TYPES, warn_unknown_entity
from wardcat.detectors.regex_detector import RegexDetector

# ── __version__ ───────────────────────────────────────────────────────────────


def test_version_exported():
    import re

    assert __version__ is not None
    # PEP 440: X.Y.Z or X.Y.ZaN / X.Y.ZbN / X.Y.ZrcN (alpha/beta/rc)
    assert re.match(r"^\d+\.\d+\.\d+", __version__), f"Invalid version: {__version__}"


# ── ReDoS Protection ──────────────────────────────────────────────────────────


class TestReDoS:
    """
    Verifies that the regex engine completes in an acceptable time
    on intensive/specially-crafted inputs. Works together with pytest-timeout
    (pyproject.toml: timeout=30) — if a regex hangs, the test is killed after 30s.
    """

    def _detector(self):
        return RegexDetector({"EMAIL", "CREDIT_CARD", "PHONE", "IBAN", "TC_ID", "IP_ADDRESS"})

    def test_long_repetitive_input(self):
        """10 KB of repetitive characters should not freeze the regex."""
        text = "a" * 10_000
        t0 = time.perf_counter()
        self._detector().detect(text)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"Regex too slow: {elapsed:.2f}s"

    def test_near_match_email_pattern(self):
        """A sequence that resembles an email but does not match should not freeze the regex."""
        text = ("a" * 50 + "@") * 200  # many incorrect @
        t0 = time.perf_counter()
        self._detector().detect(text)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"Regex too slow: {elapsed:.2f}s"

    def test_near_match_credit_card(self):
        """A sequence that resembles a card number but is invalid."""
        text = "4111 " * 2000  # repeating card prefix
        t0 = time.perf_counter()
        self._detector().detect(text)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"Regex too slow: {elapsed:.2f}s"

    def test_mixed_separators_iban(self):
        """Long sequence resembling IBAN format but with errors."""
        text = "TR" + "0" * 500
        t0 = time.perf_counter()
        self._detector().detect(text)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"Regex too slow: {elapsed:.2f}s"


# ── Large Input ───────────────────────────────────────────────────────────────


class TestLargeInput:
    def _guard(self):
        return make_legacy_guard(use_ner=False)

    def test_100kb_clean_text(self):
        """100 KB of clean text should be processable."""
        text = "Bu temiz bir metin, hassas veri yok. " * 2_800  # ~100KB
        result = self._guard().scan(text)
        assert result.is_clean

    def test_100kb_with_pii(self):
        """PII should be detectable within 100 KB of text."""
        filler = "Lorem ipsum dolor sit amet. " * 1_000
        text = filler + " email: test@example.com " + filler
        result = self._guard().scan(text)
        emails = [v for v in result.violations if v.entity_type == "EMAIL"]
        assert len(emails) >= 1

    def test_batch_many_items(self):
        """A batch of 100 items should be processable without any crash."""
        texts = [f"User {i}: user{i}@example.com" for i in range(100)]
        results = self._guard().scan_batch(texts)
        assert len(results) == 100
        assert all(any(v.entity_type == "EMAIL" for v in r.violations) for r in results)

    def test_scan_batch_large_items(self):
        """Performance check: batch of 10 items each 10 KB."""
        texts = [("temiz metin " * 500) for _ in range(10)]
        t0 = time.perf_counter()
        results = Wardcat(use_ner=False).scan_batch(texts)
        elapsed = time.perf_counter() - t0
        assert len(results) == 10
        assert elapsed < 10.0, f"Batch too slow: {elapsed:.2f}s"


# ── Thread-Safety ─────────────────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_scan_no_crash(self):
        """
        The same Wardcat instance should be callable from 10 concurrent threads.
        Results should be consistent and no exceptions should occur.
        """
        guard = Wardcat(use_ner=False)
        texts = [
            "email: test@example.com",
            "kart: 4111111111111111",
            "temiz metin",
            "TC: 12345678950",
            "IBAN TR33 0006 1005 1978 6457 8413 26",
        ]
        errors: list[Exception] = []
        results: list = [None] * (len(texts) * 2)

        def worker(idx: int, text: str):
            try:
                results[idx] = guard.scan(text)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(i, texts[i % len(texts)]))
            for i in range(len(texts) * 2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors, f"Thread errors: {errors}"
        assert all(r is not None for r in results)

    def test_concurrent_scan_batch(self):
        """scan_batch should be callable concurrently."""
        guard = Wardcat(use_ner=False)
        batch = ["a@b.com", "temiz", "4111111111111111"] * 10
        errors: list[Exception] = []

        def worker():
            try:
                guard.scan_batch(batch)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors


# ── scan_batch Error Isolation ────────────────────────────────────────────────


class TestScanBatchIsolation:
    def test_exception_in_one_item_does_not_fail_others(self):
        """
        Even if an unexpected error occurs in one item, the others should be scanned successfully.
        """
        guard = make_legacy_guard(use_ner=False)

        original_scan = guard._engine.scan

        def flaky_scan(text):
            # Fail on the credit card text so the error is deterministic regardless of thread order
            if "4111111111111111" in text:
                raise RuntimeError("Simulated error")
            return original_scan(text)

        guard._engine.scan = flaky_scan

        texts = ["a@b.com", "4111111111111111", "c@d.com"]
        results = guard.scan_batch(texts)

        assert len(results) == 3
        # 2nd item errors → original text returned, considered clean
        assert results[1].original_text == "4111111111111111"
        assert results[1].is_clean
        # Others normal
        assert any(v.entity_type == "EMAIL" for v in results[0].violations)
        assert any(v.entity_type == "EMAIL" for v in results[2].violations)

    def test_empty_batch_returns_empty(self):
        assert Wardcat(use_ner=False).scan_batch([]) == []


# ── Environment Variable Configuration ───────────────────────────────────────


class TestLibraryIgnoresEnv:
    """The library is config-explicit: load_config / Wardcat never read env vars.

    (Reading WARDCAT_* env vars is the CLI's job — see tests/unit/test_cli.py.)
    """

    def test_load_config_ignores_aiguard_salt(self):
        with patch.dict(os.environ, {"WARDCAT_SALT": "env-test-salt"}):
            cfg = load_config()
        assert cfg["salt"] == ""  # not read from env

    def test_load_config_ignores_llm_env(self):
        with patch.dict(
            os.environ,
            {"WARDCAT_LLM_URL": "http://remote:11434", "WARDCAT_LLM_MODEL": "mistral:7b"},
        ):
            cfg = load_config()
        assert cfg["llm_detector"]["base_url"] == "http://localhost:11434"
        assert cfg["llm_detector"]["model"] == "llama3.2"

    def test_aiguard_ignores_env_salt(self):
        from wardcat import Wardcat

        with patch.dict(os.environ, {"WARDCAT_SALT": "env-salt"}):
            guard = Wardcat(use_ner=False)
        assert guard._config["salt"] == ""  # explicit only

    def test_explicit_salt_wins_over_env(self):
        from wardcat import Wardcat

        with patch.dict(os.environ, {"WARDCAT_SALT": "env-salt"}):
            guard = Wardcat(use_ner=False, salt="explicit-salt")
        assert guard._config["salt"] == "explicit-salt"


# ── Config Validation ─────────────────────────────────────────────────────────


class TestConfigValidation:
    def test_invalid_action_raises(self):
        cfg = {"entities": {"EMAIL": {"enabled": True, "action": "invalid_action"}}}
        with pytest.raises(ValueError, match="Invalid action"):
            validate_config(cfg)

    def test_valid_actions_pass(self):
        cfg = {
            "entities": {
                "EMAIL": {"enabled": True, "action": "warn"},
                "TC_ID": {"enabled": True, "action": "hash"},
            },
            "llm_detector": {"backend": "ollama", "timeout": 60},
        }
        validate_config(cfg)  # should not raise

    def test_invalid_backend_raises(self):
        cfg = {
            "entities": {},
            "llm_detector": {"backend": "unknown_backend", "timeout": 60},
        }
        with pytest.raises(ValueError, match="Invalid LLM backend"):
            validate_config(cfg)

    def test_invalid_timeout_raises(self):
        cfg = {
            "entities": {},
            "llm_detector": {"backend": "ollama", "timeout": -5},
        }
        with pytest.raises(ValueError, match="Invalid.*timeout"):
            validate_config(cfg)

    def test_add_entity_invalid_action_raises(self):
        guard = Wardcat(use_ner=False)
        with pytest.raises(ValueError, match="Invalid action"):
            guard.add_entity("EMAIL", action="delete")


# ── Entity Type Warning ───────────────────────────────────────────────────────


class TestEntityTypeWarning:
    def test_known_types_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="wardcat.core.models"):
            for et in KNOWN_ENTITY_TYPES:
                warn_unknown_entity(et)
        assert "Unknown" not in caplog.text

    def test_typo_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="wardcat.core.models"):
            warn_unknown_entity("CREIDT_CARD")  # typo
        assert "CREIDT_CARD" in caplog.text

    def test_configure_unknown_entity_warns(self, caplog):
        guard = Wardcat(use_ner=False)
        with caplog.at_level(logging.WARNING, logger="wardcat.core.models"):
            guard.add_entity("TYPO_ENTITY", action="warn")
        assert "TYPO_ENTITY" in caplog.text


# ── Salt Warning ──────────────────────────────────────────────────────────────


class TestSaltWarning:
    def test_empty_salt_with_hash_action_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="wardcat.guard"):
            make_legacy_guard(use_ner=False, salt="")
        assert "WARDCAT_SALT" in caplog.text

    def test_nonempty_salt_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="wardcat.guard"):
            Wardcat(use_ner=False, salt="my-secret-salt")
        assert "WARDCAT_SALT" not in caplog.text


# ── Logging Integration ───────────────────────────────────────────────────────


class TestLogging:
    def test_engine_logs_scan_completion(self, caplog):
        from wardcat.config.loader import load_config
        from wardcat.core.engine import DetectionEngine

        cfg = load_config()
        cfg.setdefault("entities", {})["EMAIL"] = {"enabled": True, "action": "warn"}
        from wardcat.detectors.regex_detector import RegexDetector

        engine = DetectionEngine(cfg, [RegexDetector({"EMAIL"})])
        with caplog.at_level(logging.INFO, logger="wardcat.core.engine"):
            engine.scan("test@example.com")
        assert "scan completed" in caplog.text

    def test_llm_backend_error_surfaced_and_logged(self, caplog):
        from unittest.mock import MagicMock

        from wardcat.core.engine import DetectionEngine
        from wardcat.detectors.llm_detector import LLMDetector

        backend = MagicMock()
        backend.complete_messages.side_effect = ConnectionError("no connection")
        det = LLMDetector(backend=backend, enabled_entities={"EMAIL"})
        engine = DetectionEngine({"salt": "", "entities": {}}, [det])
        with caplog.at_level(logging.WARNING, logger="wardcat.core.engine"):
            result = engine.scan("test@example.com")
        # The backend failure is surfaced on the result and logged by the engine,
        # not silently swallowed by the detector.
        assert result.warnings and "did not run" in result.warnings[0]
        assert "skipped" in caplog.text.lower()

"""Surfacing degraded scans: when a detector layer cannot run (e.g. the LLM
backend is unreachable), the scan still returns the other layers' results but
records the failure on ``ScanResult.warnings`` so the caller is not misled into
thinking every layer ran.
"""

from __future__ import annotations

import asyncio

import pytest

from wardcat.core.engine import DetectionEngine
from wardcat.detectors.base import BaseDetector, DetectedSpan
from wardcat.llm.backends.base import BaseLLMBackend


class _Fixed(BaseDetector):
    def __init__(self, spans: list[DetectedSpan]) -> None:
        self._spans = spans

    def detect(self, text, candidates=None):
        return list(self._spans)


class _Raising(BaseDetector):
    def detect(self, text, candidates=None):
        raise ConnectionError("backend down")


def _engine(detectors, actions=None):
    cfg = {
        "salt": "",
        "entities": {k: {"enabled": True, "action": v} for k, v in (actions or {}).items()},
    }
    return DetectionEngine(cfg, detectors)


class TestEngineWarnings:
    def test_failure_becomes_warning_and_others_still_run(self):
        span = DetectedSpan("EMAIL", "a@b.com", 0, 7)
        eng = _engine([_Fixed([span]), _Raising()], {"EMAIL": "redact"})
        res = eng.scan("a@b.com")
        assert len(res.violations) == 1  # the working layer still produced output
        assert res.warnings, "expected a warning for the failed layer"
        assert "_Raising" in res.warnings[0]
        assert "backend down" in res.warnings[0]

    def test_no_warnings_when_all_ok(self):
        eng = _engine([_Fixed([])])
        res = eng.scan("clean text")
        assert res.warnings == []

    def test_redacted_includes_warnings(self):
        eng = _engine([_Raising()])
        red = eng.scan("x").redacted()
        assert "warnings" in red
        assert red["warnings"]

    def test_async_failure_becomes_warning(self):
        span = DetectedSpan("EMAIL", "a@b.com", 0, 7)
        eng = _engine([_Fixed([span]), _Raising()], {"EMAIL": "redact"})
        res = asyncio.run(eng.scan_async("a@b.com"))
        assert len(res.violations) == 1
        assert any("_Raising" in w for w in res.warnings)


class _DeadBackend(BaseLLMBackend):
    def complete(self, prompt, *, timeout=60):
        raise ConnectionError("no ollama")

    def complete_messages(self, messages, *, timeout=60):
        raise ConnectionError("no ollama")

    def list_models(self):
        return []

    def pull_model(self, model, *, on_progress=None):
        pass


class TestLLMDetectorPropagates:
    def test_connection_error_propagates(self):
        from wardcat.detectors.llm_detector import LLMDetector

        det = LLMDetector(_DeadBackend(), {"EMAIL"})
        with pytest.raises(ConnectionError):
            det.detect("Mail me at a@b.com please.")


class TestOrphanEntityWarning:
    """Enabling an entity whose only layer is off is a silent no-op — warn about it."""

    def test_person_without_a_model_layer_warns(self, caplog):
        import logging

        from wardcat import Entity, Wardcat

        guard = Wardcat(salt="s").add_entity(Entity.PERSON)  # needs NER or LLM, has neither
        with caplog.at_level(logging.WARNING, logger="wardcat.guard"):
            guard.scan("John Smith called.")
        assert any("PERSON" in r.message and "no active layer" in r.message for r in caplog.records)

    def test_person_covered_once_ner_is_added(self, caplog):
        import logging

        from wardcat import Entity, Wardcat

        # Same PERSON entity, but now with a NER layer → no orphan warning.
        guard = (
            Wardcat(salt="s")
            .with_ner(spacy_model="en_core_web_sm", auto_download=False)
            .add_entity(Entity.PERSON)
        )
        with caplog.at_level(logging.WARNING, logger="wardcat.guard"):
            guard.scan("John Smith called.")
        assert not any("no active layer" in r.message for r in caplog.records)

    def test_no_warning_when_layer_is_active(self, caplog):
        import logging

        from wardcat import Entity, Wardcat

        # EMAIL is a regex entity and the regex layer is always on when enabled.
        guard = Wardcat(salt="s").add_entity(Entity.EMAIL)
        with caplog.at_level(logging.WARNING, logger="wardcat.guard"):
            guard.scan("a@b.com")
        assert not any("no active layer" in r.message for r in caplog.records)

    def test_warns_once_not_every_scan(self, caplog):
        import logging

        from wardcat import Entity, Wardcat

        guard = Wardcat(salt="s").add_entity(Entity.PERSON)
        with caplog.at_level(logging.WARNING, logger="wardcat.guard"):
            guard.scan("John Smith")
            guard.scan("Jane Doe")
        orphan_warnings = [r for r in caplog.records if "no active layer" in r.message]
        assert len(orphan_warnings) == 1


class TestGuardIntegration:
    def test_scan_surfaces_llm_unavailable(self):
        from wardcat import Backend, Wardcat

        guard = (
            Wardcat(salt="s")
            .add_entity("EMAIL", "redact")
            .with_llm(backend=Backend.OLLAMA, base_url="http://127.0.0.1:59999", model="x")
        )
        res = guard.scan("Mail me at a@b.com")
        # Regex layer still redacted the email …
        assert res.sanitized_text.count("[EMAIL]") == 1
        # … and the LLM unavailability is surfaced, not silent.
        assert res.warnings
        assert any("did not run" in w for w in res.warnings)

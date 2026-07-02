"""Surfacing degraded scans: when a detector layer cannot run (e.g. the LLM
backend is unreachable), the scan still returns the other layers' results but
records the failure on ``ScanResult.warnings`` so the caller is not misled into
thinking every layer ran.
"""

from __future__ import annotations

import asyncio

import pytest

from ai_guard.core.engine import DetectionEngine
from ai_guard.detectors.base import BaseDetector, DetectedSpan
from ai_guard.llm.backends.base import BaseLLMBackend


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
        from ai_guard.detectors.llm_detector import LLMDetector

        det = LLMDetector(_DeadBackend(), {"EMAIL"})
        with pytest.raises(ConnectionError):
            det.detect("Mail me at a@b.com please.")


class TestGuardIntegration:
    def test_scan_surfaces_llm_unavailable(self):
        from ai_guard import AIGuard, Backend

        guard = (
            AIGuard(salt="s", use_ner=False)
            .add_entity("EMAIL", "redact")
            .with_llm(backend=Backend.OLLAMA, base_url="http://127.0.0.1:59999", model="x")
        )
        res = guard.scan("Mail me at a@b.com")
        # Regex layer still redacted the email …
        assert res.sanitized_text.count("[EMAIL]") == 1
        # … and the LLM unavailability is surfaced, not silent.
        assert res.warnings
        assert any("did not run" in w for w in res.warnings)

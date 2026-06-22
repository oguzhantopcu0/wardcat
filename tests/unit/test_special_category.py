"""
GDPR Article 9 special-category data (SPECIAL_CATEGORY).

LLM-only, semantic, off by default. No regex/NER pattern — only the LLM can
flag health/religion/ethnicity/political/sexual-orientation statements.
The backend is mocked; no real model calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from ai_guard.config.loader import DEFAULT_CONFIG
from ai_guard.core.engine import DetectionEngine
from ai_guard.detectors.base import BaseDetector, DetectedSpan
from ai_guard.detectors.llm_detector import LLMDetector
from ai_guard.llm.backends.base import BaseLLMBackend
from ai_guard.llm.prompt import build_messages

# ── Defaults: off by default ───────────────────────────────────────────────


class TestSpecialCategoryDefaults:
    def test_disabled_in_default_llm_entities(self):
        sc = DEFAULT_CONFIG["llm_detector"]["entities"]["SPECIAL_CATEGORY"]
        assert sc["enabled"] is False
        assert sc["action"] == "redact"

    def test_known_entity_type(self):
        from ai_guard.core.models import KNOWN_ENTITY_TYPES

        assert "SPECIAL_CATEGORY" in KNOWN_ENTITY_TYPES

    def test_not_a_regex_entity(self):
        from ai_guard.core.registry import NER_ENTITIES, REGEX_ENTITIES

        assert "SPECIAL_CATEGORY" not in REGEX_ENTITIES
        assert "SPECIAL_CATEGORY" not in NER_ENTITIES


# ── Prompt ─────────────────────────────────────────────────────────────────


class TestSpecialCategoryPrompt:
    def test_definition_present_when_requested(self):
        system = build_messages("x", {"SPECIAL_CATEGORY"})[0]["content"]
        assert "SPECIAL_CATEGORY" in system
        assert "Article 9" in system


# ── LLM detector ───────────────────────────────────────────────────────────


class TestSpecialCategoryDetection:
    def _detector(self, response: str) -> LLMDetector:
        backend = MagicMock(spec=BaseLLMBackend)
        backend.complete_messages.return_value = response
        return LLMDetector(backend=backend, enabled_entities={"SPECIAL_CATEGORY", "PERSON"})

    def test_health_statement_detected(self):
        det = self._detector('[{"type":"SPECIAL_CATEGORY","text":"HIV pozitif"}]')
        spans = det.detect("Hasta Mehmet Yılmaz HIV pozitif ve tedavi görüyor.")
        assert any(s.entity_type == "SPECIAL_CATEGORY" and s.text == "HIV pozitif" for s in spans)

    def test_disabled_type_is_filtered(self):
        # If the type is not in enabled_entities, it must be dropped even if returned.
        backend = MagicMock(spec=BaseLLMBackend)
        backend.complete_messages.return_value = (
            '[{"type":"SPECIAL_CATEGORY","text":"HIV pozitif"}]'
        )
        det = LLMDetector(backend=backend, enabled_entities={"PERSON"})
        spans = det.detect("Mehmet Yılmaz HIV pozitif.")
        assert not any(s.entity_type == "SPECIAL_CATEGORY" for s in spans)


# ── Engine: redaction ──────────────────────────────────────────────────────


class FakeDetector(BaseDetector):
    def __init__(self, spans):
        self._spans = spans

    def detect(self, text, candidates=None):
        return list(self._spans)


class TestSpecialCategoryRedaction:
    def test_redacted_in_output(self):
        text = "Mehmet HIV pozitif olarak kayıtlı."
        phrase = "HIV pozitif"
        start = text.find(phrase)
        fake = FakeDetector(
            [
                DetectedSpan(
                    "SPECIAL_CATEGORY", phrase, start, start + len(phrase), confidence=0.85
                ),
            ]
        )
        config = {
            "salt": "",
            "entities": {"SPECIAL_CATEGORY": {"enabled": True, "action": "redact"}},
        }
        result = DetectionEngine(config, [fake]).scan(text)
        assert "[SPECIAL_CATEGORY]" in result.sanitized_text
        assert phrase not in result.sanitized_text

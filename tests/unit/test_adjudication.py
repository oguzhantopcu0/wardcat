"""
Ensemble adjudication tests.

When enabled, the engine passes regex/NER candidate spans to the LLM, which
verifies/relabels/drops them and adds new PII in a single call. Deterministic
(confidence >= 1.0) spans are always kept. LLM-only deployments (no candidates)
behave exactly as pure detection.

The LLM backend is mocked — no real model calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from wardcat.core.engine import DetectionEngine
from wardcat.detectors.base import BaseDetector, DetectedSpan
from wardcat.detectors.llm_detector import LLMDetector
from wardcat.llm.backends.base import BaseLLMBackend
from wardcat.llm.prompt import build_messages

# ── Helpers ────────────────────────────────────────────────────────────────


class FakeDetector(BaseDetector):
    """Returns a fixed set of spans — stands in for regex/NER."""

    def __init__(self, spans: list[DetectedSpan]) -> None:
        self._spans = spans

    def detect(self, text, candidates=None):
        return list(self._spans)


def _mock_llm(response: str, entities: set[str]) -> LLMDetector:
    backend = MagicMock(spec=BaseLLMBackend)
    backend.complete_messages.return_value = response
    return LLMDetector(backend=backend, enabled_entities=entities)


def _engine(detectors, adjudicate: bool) -> DetectionEngine:
    config = {
        "salt": "",
        "entities": {
            "CREDIT_CARD": {"enabled": True, "action": "hash"},
            "EMAIL": {"enabled": True, "action": "warn"},
            "PERSON": {"enabled": True, "action": "hash"},
        },
        "llm_detector": {"adjudicate": adjudicate},
    }
    return DetectionEngine(config, detectors)


# ── Prompt ─────────────────────────────────────────────────────────────────


class TestAdjudicationPrompt:
    def test_candidates_add_adjudication_block(self):
        msgs = build_messages(
            "some text",
            {"EMAIL"},
            candidates=[("PERSON", "Senior Backend Engineer")],
        )
        system = msgs[0]["content"]
        assert "CANDIDATE FINDINGS" in system
        assert "Senior Backend Engineer" in system

    def test_no_candidates_no_block(self):
        msgs = build_messages("some text", {"EMAIL"})
        assert "CANDIDATE FINDINGS" not in msgs[0]["content"]

    def test_empty_candidates_no_block(self):
        msgs = build_messages("some text", {"EMAIL"}, candidates=[])
        assert "CANDIDATE FINDINGS" not in msgs[0]["content"]


# ── LLM detector candidate routing ─────────────────────────────────────────


class TestDetectorCandidates:
    def test_candidates_passed_to_backend(self):
        det = _mock_llm("[]", {"PERSON", "EMAIL"})
        text = "Senior Backend Engineer wrote a@b.com"
        cand = [DetectedSpan("PERSON", "Senior Backend Engineer", 0, 23, confidence=0.85)]
        det.detect(text, candidates=cand)
        messages = det.backend.complete_messages.call_args[0][0]
        system = messages[0]["content"]
        assert "CANDIDATE FINDINGS" in system
        assert "Senior Backend Engineer" in system

    def test_llm_only_unaffected_without_candidates(self):
        det = _mock_llm('[{"type":"EMAIL","text":"a@b.com"}]', {"EMAIL"})
        spans = det.detect("reach me at a@b.com")
        assert any(s.entity_type == "EMAIL" for s in spans)
        system = det.backend.complete_messages.call_args[0][0][0]["content"]
        assert "CANDIDATE FINDINGS" not in system


# ── Engine adjudication behaviour ──────────────────────────────────────────


class TestEngineAdjudication:
    TEXT = "Senior Backend Engineer contacted a@b.com about card 4111 1111 1111 1111"

    def _candidates(self):
        cc = "4111 1111 1111 1111"
        cc_start = self.TEXT.find(cc)
        return FakeDetector(
            [
                # NER false positive — model-based, low confidence
                DetectedSpan("PERSON", "Senior Backend Engineer", 0, 23, confidence=0.85),
                # Regex deterministic — must always survive
                DetectedSpan("CREDIT_CARD", cc, cc_start, cc_start + len(cc), confidence=1.0),
            ]
        )

    def test_llm_drops_ner_false_positive(self):
        # LLM omits the PERSON candidate and adds a new EMAIL.
        llm = _mock_llm('[{"type":"EMAIL","text":"a@b.com"}]', {"PERSON", "EMAIL", "CREDIT_CARD"})
        engine = _engine([self._candidates(), llm], adjudicate=True)
        result = engine.scan(self.TEXT)
        types = {v.entity_type for v in result.violations}
        assert "PERSON" not in types  # dropped by adjudication
        assert "CREDIT_CARD" in types  # deterministic, kept
        assert "EMAIL" in types  # new LLM find
        assert "Senior Backend Engineer" in result.sanitized_text

    def test_deterministic_kept_even_if_llm_silent(self):
        # LLM returns nothing — the regex card must still be protected.
        llm = _mock_llm("[]", {"PERSON", "EMAIL", "CREDIT_CARD"})
        engine = _engine([self._candidates(), llm], adjudicate=True)
        result = engine.scan(self.TEXT)
        types = {v.entity_type for v in result.violations}
        assert "CREDIT_CARD" in types
        assert "PERSON" not in types

    def test_all_regex_tiers_kept_model_overridable(self):
        # Every regex tier (fuzzy ADDRESS 0.90, structural EMAIL 0.97, checksum
        # CREDIT_CARD 1.0) is protected in adjudication; only a model-based
        # candidate (PERSON 0.85) is dropped when the LLM is silent. For a PII
        # tool, never letting the LLM drop a deterministic match avoids leaks.
        from wardcat.core.engine import DetectionEngine
        from wardcat.detectors.regex_detector import CONF_CHECKSUM, CONF_FUZZY, CONF_STRUCTURAL

        text = "Adres Bağdat Caddesi, mail a@b.com, kart 4111 1111 1111 1111, kişi Ali Veli"

        def _span(etype, val, conf):
            i = text.find(val)
            return DetectedSpan(etype, val, i, i + len(val), confidence=conf)

        cands = FakeDetector(
            [
                _span("ADDRESS", "Bağdat Caddesi", CONF_FUZZY),
                _span("EMAIL", "a@b.com", CONF_STRUCTURAL),
                _span("CREDIT_CARD", "4111 1111 1111 1111", CONF_CHECKSUM),
                _span("PERSON", "Ali Veli", 0.85),  # model-based
            ]
        )
        llm = _mock_llm("[]", {"ADDRESS", "EMAIL", "CREDIT_CARD", "PERSON"})  # silent
        config = {
            "salt": "",
            "entities": {
                e: {"enabled": True, "action": "redact"}
                for e in ("ADDRESS", "EMAIL", "CREDIT_CARD", "PERSON")
            },
            "llm_detector": {"adjudicate": True},
        }
        types = {v.entity_type for v in DetectionEngine(config, [cands, llm]).scan(text).violations}
        assert {"ADDRESS", "EMAIL", "CREDIT_CARD"} <= types  # all regex tiers kept
        assert "PERSON" not in types  # model-based — silent LLM dropped it

    def test_union_mode_keeps_ner_false_positive(self):
        # With adjudication OFF, the NER false positive survives (current behaviour).
        llm = _mock_llm("[]", {"PERSON", "EMAIL", "CREDIT_CARD"})
        engine = _engine([self._candidates(), llm], adjudicate=False)
        result = engine.scan(self.TEXT)
        types = {v.entity_type for v in result.violations}
        assert "PERSON" in types
        assert "CREDIT_CARD" in types


def test_date_of_birth_is_an_llm_entity():
    # DATE_OF_BIRTH must be a supported LLM entity so the LLM layer can catch
    # birth dates the regex misses (e.g. "14.03.1985 doğumlu").
    from wardcat.llm.prompt import SUPPORTED_ENTITIES

    assert "DATE_OF_BIRTH" in SUPPORTED_ENTITIES

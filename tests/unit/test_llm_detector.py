"""
LLMDetector unit tests.

No real LLM calls are made; the backend is mocked.
This allows tests to run without Ollama installed.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ai_guard.detectors.llm_detector import LLMDetector
from ai_guard.llm.backends.base import BaseLLMBackend
from ai_guard.llm.prompt import build_prompt


# ── Mock backend helper ──────────────────────────────────────────────────────

def _mock_backend(response: str) -> BaseLLMBackend:
    backend = MagicMock(spec=BaseLLMBackend)
    backend.complete.return_value = response
    backend.complete_messages.return_value = response
    return backend


def _detector(response: str, entities: set[str] | None = None) -> LLMDetector:
    if entities is None:
        entities = {"CREDIT_CARD", "EMAIL", "PERSON", "TC_ID", "IBAN", "PHONE"}
    return LLMDetector(
        backend=_mock_backend(response),
        enabled_entities=entities,
    )


# ═══════════════════════════════════════════════════════════════════════════
# JSON response parsing
# ═══════════════════════════════════════════════════════════════════════════

class TestParseResponse:
    def test_clean_json_array(self):
        det = _detector('[{"type":"EMAIL","text":"a@b.com"}]')
        spans = det.detect("a@b.com")
        assert len(spans) == 1
        assert spans[0].entity_type == "EMAIL"
        assert spans[0].text == "a@b.com"

    def test_empty_json_array(self):
        det = _detector("[]")
        spans = det.detect("temiz metin")
        assert spans == []

    def test_markdown_code_block_stripped(self):
        response = '```json\n[{"type":"PERSON","text":"Ali Veli"}]\n```'
        det = _detector(response, {"PERSON"})
        spans = det.detect("Ali Veli burada.")
        assert any(s.entity_type == "PERSON" for s in spans)

    def test_json_with_surrounding_text(self):
        """Small models may add explanation text."""
        response = 'Tespit ettim:\n[{"type":"EMAIL","text":"x@y.com"}]\nBaşka bir şey yok.'
        det = _detector(response)
        spans = det.detect("x@y.com")
        assert len(spans) == 1

    def test_malformed_json_returns_empty(self):
        det = _detector("Bu JSON değil, sadece metin.")
        spans = det.detect("kart: 4111111111111111")
        assert spans == []

    def test_truncated_json_returns_empty(self):
        det = _detector('[{"type":"EMAIL","text":"x@y.co')  # truncated
        spans = det.detect("x@y.com")
        assert spans == []

    def test_json_object_instead_of_array_returns_empty(self):
        det = _detector('{"type":"EMAIL","text":"x@y.com"}')
        spans = det.detect("x@y.com")
        assert spans == []


# ═══════════════════════════════════════════════════════════════════════════
# Span location
# ═══════════════════════════════════════════════════════════════════════════

class TestSpanLocation:
    def test_correct_start_end(self):
        text = "bana a@b.com yaz"
        det = _detector('[{"type":"EMAIL","text":"a@b.com"}]')
        spans = det.detect(text)
        assert spans[0].start == 5
        assert spans[0].end   == 12
        assert text[spans[0].start:spans[0].end] == "a@b.com"

    def test_multiple_occurrences_all_located(self):
        text = "a@b.com ve a@b.com"
        det = _detector('[{"type":"EMAIL","text":"a@b.com"}]')
        spans = det.detect(text)
        assert len(spans) == 2
        assert spans[0].start == 0
        assert spans[1].start == 11

    def test_entity_not_in_text_skipped(self):
        """When the LLM hallucinates (text not in input), it should be skipped."""
        det = _detector('[{"type":"EMAIL","text":"hayali@yok.com"}]')
        spans = det.detect("başka bir metin")
        assert spans == []

    def test_multiple_entity_types(self):
        text = "TC: 12345678950 kart: 4111111111111111"
        response = json.dumps([
            {"type": "TC_ID",       "text": "12345678950"},
            {"type": "CREDIT_CARD", "text": "4111111111111111"},
        ])
        det = _detector(response)
        spans = det.detect(text)
        types = {s.entity_type for s in spans}
        assert "TC_ID"       in types
        assert "CREDIT_CARD" in types

    def test_no_duplicate_spans_for_same_position(self):
        """Even if the same entity is returned twice, only one span should be created."""
        response = json.dumps([
            {"type": "EMAIL", "text": "a@b.com"},
            {"type": "EMAIL", "text": "a@b.com"},   # duplicate
        ])
        det = _detector(response)
        spans = det.detect("a@b.com")
        assert len(spans) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Entity filter
# ═══════════════════════════════════════════════════════════════════════════

class TestEntityFilter:
    def test_disabled_entity_skipped(self):
        """Even if the LLM returns a disabled entity, no span should be created."""
        det = _detector(
            '[{"type":"ORG","text":"Apple"}]',
            entities={"PERSON"},   # ORG disabled
        )
        spans = det.detect("Apple şirketi")
        assert not any(s.entity_type == "ORG" for s in spans)

    def test_unknown_entity_type_skipped(self):
        det = _detector(
            '[{"type":"BILINMEYEN","text":"gizli"}]',
            entities={"EMAIL"},
        )
        spans = det.detect("gizli metin")
        assert spans == []

    def test_entity_type_normalized_to_uppercase(self):
        """Even if the LLM returns lowercase, the entity type should be normalized."""
        det = _detector(
            '[{"type":"email","text":"a@b.com"}]',
            entities={"EMAIL"},
        )
        spans = det.detect("a@b.com")
        # lowercase "email" → is "EMAIL" in enabled_entities?
        # upper() is applied inside _locate_spans → should match
        assert len(spans) == 1
        assert spans[0].entity_type == "EMAIL"


# ═══════════════════════════════════════════════════════════════════════════
# Error handling
# ═══════════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    def test_connection_error_returns_empty_with_warning(self, caplog):
        import logging
        backend = MagicMock(spec=BaseLLMBackend)
        backend.complete_messages.side_effect = ConnectionError("Connection refused")
        det = LLMDetector(backend=backend, enabled_entities={"EMAIL"})

        with caplog.at_level(logging.WARNING, logger="ai_guard.detectors.llm_detector"):
            spans = det.detect("a@b.com")

        assert spans == []
        assert any("connection error" in r.message.lower() for r in caplog.records)

    def test_generic_exception_returns_empty_with_warning(self, caplog):
        import logging
        backend = MagicMock(spec=BaseLLMBackend)
        backend.complete_messages.side_effect = RuntimeError("Beklenmedik hata")
        det = LLMDetector(backend=backend, enabled_entities={"EMAIL"})

        with caplog.at_level(logging.WARNING, logger="ai_guard.detectors.llm_detector"):
            spans = det.detect("a@b.com")

        assert spans == []
        assert len(caplog.records) >= 1

    def test_empty_text_not_sent_to_backend(self):
        backend = MagicMock(spec=BaseLLMBackend)
        det = LLMDetector(backend=backend, enabled_entities={"EMAIL"})
        det.detect("")
        det.detect("   ")
        backend.complete_messages.assert_not_called()

    def test_timeout_passed_to_backend(self):
        backend = MagicMock(spec=BaseLLMBackend)
        backend.complete_messages.return_value = "[]"
        det = LLMDetector(backend=backend, enabled_entities={"EMAIL"}, timeout=120)
        det.detect("test")
        backend.complete_messages.assert_called_once()
        _, kwargs = backend.complete_messages.call_args
        assert kwargs.get("timeout") == 120


# ═══════════════════════════════════════════════════════════════════════════
# Prompt building
# ═══════════════════════════════════════════════════════════════════════════

class TestPromptBuilding:
    def test_prompt_contains_text(self):
        prompt = build_prompt("gizli metin burada", {"EMAIL"})
        assert "gizli metin burada" in prompt

    def test_prompt_contains_entity_types(self):
        prompt = build_prompt("test", {"EMAIL", "CREDIT_CARD", "PERSON"})
        assert "EMAIL"       in prompt
        assert "CREDIT_CARD" in prompt
        assert "PERSON"      in prompt

    def test_prompt_requests_json_output(self):
        prompt = build_prompt("test", {"EMAIL"})
        assert "JSON" in prompt.upper()

    def test_backend_receives_built_prompt(self):
        backend = MagicMock(spec=BaseLLMBackend)
        backend.complete_messages.return_value = "[]"
        det = LLMDetector(backend=backend, enabled_entities={"EMAIL"})
        det.detect("a@b.com")
        # complete_messages receives a list of dicts; check content inside them
        messages = backend.complete_messages.call_args[0][0]
        combined = " ".join(m["content"] for m in messages)
        assert "a@b.com" in combined
        assert "EMAIL"   in combined

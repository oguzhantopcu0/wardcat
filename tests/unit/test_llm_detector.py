"""
LLMDetector unit tests.

No real LLM calls are made; the backend is mocked.
This allows tests to run without Ollama installed.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

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
        assert spans[0].end == 12
        assert text[spans[0].start : spans[0].end] == "a@b.com"

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
        response = json.dumps(
            [
                {"type": "TC_ID", "text": "12345678950"},
                {"type": "CREDIT_CARD", "text": "4111111111111111"},
            ]
        )
        det = _detector(response)
        spans = det.detect(text)
        types = {s.entity_type for s in spans}
        assert "TC_ID" in types
        assert "CREDIT_CARD" in types

    def test_no_duplicate_spans_for_same_position(self):
        """Even if the same entity is returned twice, only one span should be created."""
        response = json.dumps(
            [
                {"type": "EMAIL", "text": "a@b.com"},
                {"type": "EMAIL", "text": "a@b.com"},  # duplicate
            ]
        )
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
            entities={"PERSON"},  # ORG disabled
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
    def test_connection_error_propagates(self):
        # A backend-unreachable error propagates so the engine can surface it on
        # the result (ScanResult.warnings), instead of being silently swallowed.
        backend = MagicMock(spec=BaseLLMBackend)
        backend.complete_messages.side_effect = ConnectionError("Connection refused")
        det = LLMDetector(backend=backend, enabled_entities={"EMAIL"})

        with pytest.raises(ConnectionError):
            det.detect("a@b.com")

    def test_generic_exception_returns_empty_with_warning(self, caplog):
        import logging

        backend = MagicMock(spec=BaseLLMBackend)
        backend.complete_messages.side_effect = OSError("Beklenmedik hata")
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


class TestHallucinationFilter:
    def test_person_single_word_rejected(self):
        """PERSON with a single word (no space) fails the format validator → not returned."""
        det = _detector('[{"type":"PERSON","text":"customer"}]', entities={"PERSON"})
        spans = det.detect("customer order placed")
        assert not any(s.entity_type == "PERSON" for s in spans)

    def test_tc_id_wrong_format_rejected(self):
        """TC_ID that doesn't match \\d{11} fails format validation → not returned."""
        det = _detector('[{"type":"TC_ID","text":"abc"}]', entities={"TC_ID"})
        spans = det.detect("abc 12345678950")
        assert not any(s.entity_type == "TC_ID" and s.text == "abc" for s in spans)

    def test_valid_person_two_words_accepted(self):
        """PERSON with two words passes the format validator → returned."""
        det = _detector('[{"type":"PERSON","text":"Ali Veli"}]', entities={"PERSON"})
        spans = det.detect("Ali Veli burada.")
        assert any(s.entity_type == "PERSON" for s in spans)

    def test_json_decode_error_returns_empty(self):
        """Response containing [invalid json] triggers JSONDecodeError → []."""
        det = _detector("[not valid json content]")
        spans = det.detect("some text")
        assert spans == []

    def test_json_non_list_returns_empty(self):
        """Response where the matched [...] contains non-JSON → []."""
        # _JSON_RE matches first [...]. If that content is invalid, JSONDecodeError → []
        det = _detector("[{broken json here}]")
        spans = det.detect("some text")
        assert spans == []


class TestCache:
    def test_cache_disabled_by_default(self):
        """With default cache_ttl=0, cache is not used."""
        backend = MagicMock()
        backend.complete_messages.return_value = "[]"
        det = LLMDetector(backend=backend, enabled_entities={"EMAIL"})
        det.detect("text1")
        det.detect("text1")
        assert backend.complete_messages.call_count == 2  # called twice (no cache)

    def test_cache_hit_returns_cached_spans(self):
        """Second call with same text returns cached spans without calling backend."""
        backend = MagicMock()
        backend.complete_messages.return_value = '[{"type":"EMAIL","text":"a@b.com"}]'
        det = LLMDetector(backend=backend, enabled_entities={"EMAIL"}, cache_ttl=60)
        spans1 = det.detect("a@b.com here")
        spans2 = det.detect("a@b.com here")
        assert backend.complete_messages.call_count == 1  # second call used cache
        assert len(spans1) == len(spans2) == 1

    def test_cache_miss_calls_backend(self):
        """Different texts get different cache entries, each calls backend."""
        backend = MagicMock()
        backend.complete_messages.return_value = "[]"
        det = LLMDetector(backend=backend, enabled_entities={"EMAIL"}, cache_ttl=60)
        det.detect("text one")
        det.detect("text two")
        assert backend.complete_messages.call_count == 2

    def test_cache_expiry_calls_backend_again(self):
        """After TTL expires, backend is called again."""
        import time

        backend = MagicMock()
        backend.complete_messages.return_value = "[]"
        det = LLMDetector(backend=backend, enabled_entities={"EMAIL"}, cache_ttl=0.01)
        det.detect("text")
        time.sleep(0.05)  # wait for cache to expire
        det.detect("text")
        assert backend.complete_messages.call_count == 2

    def test_cache_stores_spans(self):
        """Cached spans are identical to original detection result."""
        backend = MagicMock()
        backend.complete_messages.return_value = '[{"type":"EMAIL","text":"a@b.com"}]'
        det = LLMDetector(backend=backend, enabled_entities={"EMAIL"}, cache_ttl=60)
        spans_first = det.detect("contact a@b.com please")
        spans_second = det.detect("contact a@b.com please")
        assert spans_first[0].text == spans_second[0].text
        assert spans_first[0].start == spans_second[0].start


class TestPromptBuilding:
    def test_prompt_contains_text(self):
        prompt = build_prompt("gizli metin burada", {"EMAIL"})
        assert "gizli metin burada" in prompt

    def test_prompt_contains_entity_types(self):
        prompt = build_prompt("test", {"EMAIL", "CREDIT_CARD", "PERSON"})
        assert "EMAIL" in prompt
        assert "CREDIT_CARD" in prompt
        assert "PERSON" in prompt

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
        assert "EMAIL" in combined


# ── detect_async ─────────────────────────────────────────────────────────────

import asyncio
from unittest.mock import AsyncMock


class TestDetectAsync:
    def _async_backend(self, response: str):
        backend = MagicMock(spec=BaseLLMBackend)
        backend.complete_messages.return_value = response
        backend.complete_messages_async = AsyncMock(return_value=response)
        return backend

    def test_detect_async_returns_spans(self):
        payload = json.dumps([{"type": "EMAIL", "text": "a@b.com"}])
        backend = self._async_backend(payload)
        det = LLMDetector(backend=backend, enabled_entities={"EMAIL"})
        spans = asyncio.run(det.detect_async("email: a@b.com"))
        assert any(s.entity_type == "EMAIL" for s in spans)

    def test_detect_async_empty_text_returns_empty(self):
        backend = self._async_backend("[]")
        det = LLMDetector(backend=backend, enabled_entities={"EMAIL"})
        spans = asyncio.run(det.detect_async("   "))
        assert spans == []

    def test_detect_async_cache_hit(self):
        payload = json.dumps([{"type": "EMAIL", "text": "a@b.com"}])
        backend = self._async_backend(payload)
        det = LLMDetector(backend=backend, enabled_entities={"EMAIL"}, cache_ttl=60)
        # First call — populates cache
        asyncio.run(det.detect_async("email: a@b.com"))
        # Second call — should use cache; complete_messages_async called only once
        asyncio.run(det.detect_async("email: a@b.com"))
        assert backend.complete_messages_async.call_count == 1

    def test_detect_async_connection_error_propagates(self):
        backend = MagicMock(spec=BaseLLMBackend)
        backend.complete_messages_async = AsyncMock(side_effect=ConnectionError("refused"))
        det = LLMDetector(backend=backend, enabled_entities={"EMAIL"})
        with pytest.raises(ConnectionError):
            asyncio.run(det.detect_async("a@b.com"))

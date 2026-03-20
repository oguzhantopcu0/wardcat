"""
LLM detector integration tests.

Tests the cooperation of LLMGuard + LLMDetector via a mock backend,
without a real Ollama service.
"""
from __future__ import annotations

import json
import warnings
from unittest.mock import MagicMock

import pytest

from ai_guard import LLMGuard
from ai_guard.core.models import Action
from ai_guard.detectors.llm_detector import LLMDetector
from ai_guard.llm.backends.base import BaseLLMBackend


def _mock_backend(response: str) -> BaseLLMBackend:
    b = MagicMock(spec=BaseLLMBackend)
    b.complete.return_value = response
    b.complete_messages.return_value = response
    return b


def _guard_with_llm(response: str, entities: set[str] | None = None) -> LLMGuard:
    """
    Returns an LLMGuard with a mocked LLM backend.
    Instead of patching LLMGuard's _build_llm_detector directly,
    we inject the detector afterwards.
    """
    guard = LLMGuard(use_ner=False, use_llm=False)  # build without LLM first
    enabled = entities or {"CREDIT_CARD", "EMAIL", "PERSON", "TC_ID", "IBAN",
                           "PHONE", "IP_ADDRESS", "ADDRESS", "CUSTOM_SECRET"}
    llm_det = LLMDetector(
        backend=_mock_backend(response),
        enabled_entities=enabled,
    )
    # Add LLM entities to engine config (avoid warn if not present when guard._rebuild() runs)
    for e in enabled:
        guard._config["entities"].setdefault(e, {"enabled": True, "action": "warn"})
    guard._config["entities"]["PERSON"]        = {"enabled": True, "action": "hash"}
    guard._config["entities"]["CUSTOM_SECRET"] = {"enabled": True, "action": "hash"}
    guard._detectors.append(llm_det)
    from ai_guard.core.engine import DetectionEngine
    guard._engine = DetectionEngine(guard._config, guard._detectors)
    return guard


# ═══════════════════════════════════════════════════════════════════════════
# LLM single detection
# ═══════════════════════════════════════════════════════════════════════════

class TestLLMSingleDetection:
    def test_person_detected_and_hashed(self):
        response = json.dumps([{"type": "PERSON", "text": "Ali Veli"}])
        guard = _guard_with_llm(response)
        result = guard.scan("Müşteri: Ali Veli, memnun kaldı.")
        persons = [v for v in result.violations if v.entity_type == "PERSON"]
        assert len(persons) == 1
        assert persons[0].action == Action.HASH
        assert "Ali Veli" not in result.sanitized_text
        assert "[PERSON:" in result.sanitized_text

    def test_custom_secret_hashed(self):
        response = json.dumps([{"type": "CUSTOM_SECRET", "text": "gizli-proje-kodu"}])
        guard = _guard_with_llm(response, {"CUSTOM_SECRET"})
        result = guard.scan("Proje kodu: gizli-proje-kodu")
        assert "gizli-proje-kodu" not in result.sanitized_text
        assert "[CUSTOM_SECRET:" in result.sanitized_text

    def test_email_warned(self):
        response = json.dumps([{"type": "EMAIL", "text": "ali@sirket.com"}])
        guard = _guard_with_llm(response)
        result = guard.scan("İletişim: ali@sirket.com")
        emails = [v for v in result.violations if v.entity_type == "EMAIL"]
        assert len(emails) >= 1
        assert all(v.action == Action.WARN for v in emails)


# ═══════════════════════════════════════════════════════════════════════════
# LLM + Regex hybrid
# ═══════════════════════════════════════════════════════════════════════════

class TestLLMPlusRegex:
    def test_regex_and_llm_both_detect(self):
        """Regex detects the card, LLM detects the person."""
        llm_response = json.dumps([{"type": "PERSON", "text": "Mehmet Demir"}])
        guard = _guard_with_llm(llm_response)
        text  = "Mehmet Demir kartıyla 4111111111111111 ödedi."
        result = guard.scan(text)
        types = {v.entity_type for v in result.violations}
        assert "PERSON"      in types
        assert "CREDIT_CARD" in types

    def test_llm_and_regex_same_entity_no_duplicate(self):
        """If both detectors capture the same span, overlap resolution should not duplicate it."""
        # Email can be captured by both regex and LLM
        llm_response = json.dumps([{"type": "EMAIL", "text": "a@b.com"}])
        guard = _guard_with_llm(llm_response)
        result = guard.scan("a@b.com")
        emails = [v for v in result.violations if v.entity_type == "EMAIL"]
        assert len(emails) == 1

    def test_llm_detects_contextual_pii_regex_misses(self):
        """Data that is contextually sensitive but does not match a regex pattern."""
        secret = "PROJE-ALPHA-GIZLI"
        llm_response = json.dumps([{"type": "CUSTOM_SECRET", "text": secret}])
        guard = _guard_with_llm(llm_response, {"CUSTOM_SECRET"})
        # Verify regex does not catch this first
        regex_only = LLMGuard(use_ner=False)
        assert regex_only.scan(f"kod: {secret}").is_clean

        # LLM should catch it
        result = guard.scan(f"kod: {secret}")
        assert not result.is_clean
        assert secret not in result.sanitized_text


# ═══════════════════════════════════════════════════════════════════════════
# Sanitized text integrity
# ═══════════════════════════════════════════════════════════════════════════

class TestLLMSanitizedIntegrity:
    def test_person_position_correct_in_original(self):
        name = "Ayşe Kaya"
        text = f"Müşteri adı: {name}, şikayeti var."
        llm_response = json.dumps([{"type": "PERSON", "text": name}])
        guard = _guard_with_llm(llm_response)
        result = guard.scan(text)

        for v in result.violations:
            if v.entity_type == "PERSON":
                assert text[v.start:v.end] == v.original

    def test_multiple_llm_detections_sanitized_correctly(self):
        text = "Kişi: Ali Yılmaz, kod: PROJE-X-42"
        llm_response = json.dumps([
            {"type": "PERSON",        "text": "Ali Yılmaz"},
            {"type": "CUSTOM_SECRET", "text": "PROJE-X-42"},
        ])
        guard = _guard_with_llm(llm_response, {"PERSON", "CUSTOM_SECRET"})
        result = guard.scan(text)
        assert "Ali Yılmaz" not in result.sanitized_text
        assert "PROJE-X-42" not in result.sanitized_text
        assert "[PERSON:"        in result.sanitized_text
        assert "[CUSTOM_SECRET:" in result.sanitized_text


# ═══════════════════════════════════════════════════════════════════════════
# Error handling — even if LLM errors, other detectors should run
# ═══════════════════════════════════════════════════════════════════════════

class TestLLMFaultTolerance:
    def test_llm_error_doesnt_block_regex(self):
        """Even with an LLM connection error, the regex detector should still work."""
        failing_backend = MagicMock(spec=BaseLLMBackend)
        failing_backend.complete.side_effect = ConnectionError("Ollama çevrimdışı")

        guard = LLMGuard(use_ner=False)
        llm_det = LLMDetector(backend=failing_backend, enabled_entities={"PERSON"})
        guard._detectors.append(llm_det)
        from ai_guard.core.engine import DetectionEngine
        guard._engine = DetectionEngine(guard._config, guard._detectors)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = guard.scan("kart: 4111111111111111")

        # Regex should still work
        assert any(v.entity_type == "CREDIT_CARD" for v in result.violations)

    def test_llm_empty_response_doesnt_block(self):
        guard = _guard_with_llm("[]")
        result = guard.scan("kart: 4111111111111111")
        # Regex should work
        assert any(v.entity_type == "CREDIT_CARD" for v in result.violations)

    def test_llm_malformed_response_doesnt_block(self):
        guard = _guard_with_llm("Bu JSON değil!!!")
        result = guard.scan("TC: 12345678950")
        assert any(v.entity_type == "TC_ID" for v in result.violations)


# ═══════════════════════════════════════════════════════════════════════════
# LLMGuard configuration — use_llm flag
# ═══════════════════════════════════════════════════════════════════════════

class TestLLMGuardConfig:
    def test_use_llm_false_no_llm_detector(self):
        guard = LLMGuard(use_ner=False, use_llm=False)
        from ai_guard.detectors.llm_detector import LLMDetector
        assert not any(isinstance(d, LLMDetector) for d in guard._detectors)

    def test_llm_config_stored_correctly(self):
        guard = LLMGuard(
            use_ner=False,
            use_llm=False,           # test without service; check config
            llm_model="mistral",
            llm_base_url="http://10.0.0.5:11434",
        )
        cfg = guard._config["llm_detector"]
        assert cfg["model"]    == "mistral"
        assert cfg["base_url"] == "http://10.0.0.5:11434"

    def test_invalid_backend_raises(self):
        guard = LLMGuard(use_ner=False)
        guard._config["llm_detector"]["enabled"] = True
        guard._config["llm_detector"]["backend"] = "bilinmeyen_backend"
        with pytest.raises(ValueError, match="bilinmeyen_backend"):
            guard._build_llm_detector(guard._config["llm_detector"])

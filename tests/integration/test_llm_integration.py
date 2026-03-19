"""
LLM dedektörü entegrasyon testleri.

Gerçek Ollama servisi olmadan, LLMGuard + LLMDetector işbirliğini
mock backend aracılığıyla test eder.
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
    return b


def _guard_with_llm(response: str, entities: set[str] | None = None) -> LLMGuard:
    """
    LLM backend mock'lanmış LLMGuard döndürür.
    LLMGuard'ın _build_llm_detector'ını doğrudan patch etmek yerine
    dedektörü sonradan enjekte ederiz.
    """
    guard = LLMGuard(use_ner=False, use_llm=False)  # önce LLM olmadan kur
    enabled = entities or {"CREDIT_CARD", "EMAIL", "PERSON", "TC_ID", "IBAN",
                           "PHONE", "IP_ADDRESS", "ADDRESS", "CUSTOM_SECRET"}
    llm_det = LLMDetector(
        backend=_mock_backend(response),
        enabled_entities=enabled,
    )
    # Engine config'e LLM entity'lerini ekle (guard._rebuild() yaptığında yoksa warn)
    for e in enabled:
        guard._config["entities"].setdefault(e, {"enabled": True, "action": "warn"})
    guard._config["entities"]["PERSON"]        = {"enabled": True, "action": "hash"}
    guard._config["entities"]["CUSTOM_SECRET"] = {"enabled": True, "action": "hash"}
    guard._detectors.append(llm_det)
    from ai_guard.core.engine import DetectionEngine
    guard._engine = DetectionEngine(guard._config, guard._detectors)
    return guard


# ═══════════════════════════════════════════════════════════════════════════
# LLM tekil tespit
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
# LLM + Regex hibrit
# ═══════════════════════════════════════════════════════════════════════════

class TestLLMPlusRegex:
    def test_regex_and_llm_both_detect(self):
        """Regex kartı, LLM kişiyi tespit eder."""
        llm_response = json.dumps([{"type": "PERSON", "text": "Mehmet Demir"}])
        guard = _guard_with_llm(llm_response)
        text  = "Mehmet Demir kartıyla 4111111111111111 ödedi."
        result = guard.scan(text)
        types = {v.entity_type for v in result.violations}
        assert "PERSON"      in types
        assert "CREDIT_CARD" in types

    def test_llm_and_regex_same_entity_no_duplicate(self):
        """Her iki dedektör aynı span'i yakalasa overlap çözümü tekrarlamaz."""
        # Email hem regex hem LLM tarafından yakalanabilir
        llm_response = json.dumps([{"type": "EMAIL", "text": "a@b.com"}])
        guard = _guard_with_llm(llm_response)
        result = guard.scan("a@b.com")
        emails = [v for v in result.violations if v.entity_type == "EMAIL"]
        assert len(emails) == 1

    def test_llm_detects_contextual_pii_regex_misses(self):
        """Regex kalıba uymayan ama bağlamsal olarak hassas olan veriler."""
        secret = "PROJE-ALPHA-GIZLI"
        llm_response = json.dumps([{"type": "CUSTOM_SECRET", "text": secret}])
        guard = _guard_with_llm(llm_response, {"CUSTOM_SECRET"})
        # Önce regex'in bunu yakalamadığını doğrula
        regex_only = LLMGuard(use_ner=False)
        assert regex_only.scan(f"kod: {secret}").is_clean

        # LLM ile yakalanmalı
        result = guard.scan(f"kod: {secret}")
        assert not result.is_clean
        assert secret not in result.sanitized_text


# ═══════════════════════════════════════════════════════════════════════════
# Sanitized metin bütünlüğü
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
# Hata yönetimi — LLM hata verse diğer dedektörler çalışmalı
# ═══════════════════════════════════════════════════════════════════════════

class TestLLMFaultTolerance:
    def test_llm_error_doesnt_block_regex(self):
        """LLM bağlantı hatası olsa bile regex dedektörü çalışmalı."""
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

        # Regex hâlâ çalışmalı
        assert any(v.entity_type == "CREDIT_CARD" for v in result.violations)

    def test_llm_empty_response_doesnt_block(self):
        guard = _guard_with_llm("[]")
        result = guard.scan("kart: 4111111111111111")
        # Regex çalışmalı
        assert any(v.entity_type == "CREDIT_CARD" for v in result.violations)

    def test_llm_malformed_response_doesnt_block(self):
        guard = _guard_with_llm("Bu JSON değil!!!")
        result = guard.scan("TC: 12345678901")
        assert any(v.entity_type == "TC_ID" for v in result.violations)


# ═══════════════════════════════════════════════════════════════════════════
# LLMGuard konfigürasyon — use_llm flag'i
# ═══════════════════════════════════════════════════════════════════════════

class TestLLMGuardConfig:
    def test_use_llm_false_no_llm_detector(self):
        guard = LLMGuard(use_ner=False, use_llm=False)
        from ai_guard.detectors.llm_detector import LLMDetector
        assert not any(isinstance(d, LLMDetector) for d in guard._detectors)

    def test_llm_config_stored_correctly(self):
        guard = LLMGuard(
            use_ner=False,
            use_llm=False,           # servis olmadan test; config'i kontrol et
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

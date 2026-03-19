"""
SpaCy NER dedektörü testleri (en_core_web_sm gerektirir).

pytest -m ner  →  yalnızca bu dosyayı çalıştır
"""
from __future__ import annotations

import warnings

import pytest

from ai_guard import LLMGuard
from ai_guard.detectors.ner_detector import NERDetector

pytestmark = pytest.mark.ner   # etiket: uv run pytest -m ner


@pytest.fixture(scope="module")
def ner():
    """Modül başına bir kez yükle (SpaCy ağır)."""
    return NERDetector({"PERSON", "ORG", "ADDRESS"}, model="en_core_web_sm")


@pytest.fixture(scope="module")
def guard_with_ner():
    return LLMGuard(use_ner=True, spacy_model="en_core_web_sm")


# ── PERSON tespiti ───────────────────────────────────────────────────────────

class TestPersonDetection:
    def test_full_name_detected(self, ner):
        spans = ner.detect("John Smith called us today.")
        assert any(s.entity_type == "PERSON" and "John" in s.text for s in spans)

    def test_titled_name_detected(self, ner):
        spans = ner.detect("Dr. Emily Johnson reviewed the case.")
        persons = [s.text for s in spans if s.entity_type == "PERSON"]
        assert any("Emily" in p or "Johnson" in p for p in persons)

    def test_multiple_persons(self, ner):
        spans = ner.detect("Alice met Bob and Charlie at the office.")
        persons = [s for s in spans if s.entity_type == "PERSON"]
        assert len(persons) >= 2

    def test_person_not_in_clean_text(self, ner):
        spans = ner.detect("The weather is nice today.")
        assert not any(s.entity_type == "PERSON" for s in spans)


# ── ORG tespiti ──────────────────────────────────────────────────────────────

class TestOrgDetection:
    def test_known_company(self, ner):
        spans = ner.detect("Microsoft announced new features.")
        assert any(s.entity_type == "ORG" for s in spans)

    def test_person_and_org_together(self, ner):
        spans = ner.detect("Tim Cook is the CEO of Apple.")
        types = {s.entity_type for s in spans}
        assert "PERSON" in types
        assert "ORG" in types


# ── ADDRESS tespiti (GPE / LOC → ADDRESS) ────────────────────────────────────

class TestAddressDetection:
    def test_city_detected_as_address(self, ner):
        spans = ner.detect("She lives in New York.")
        assert any(s.entity_type == "ADDRESS" for s in spans)

    def test_country_detected_as_address(self, ner):
        spans = ner.detect("The office is located in Germany.")
        assert any(s.entity_type == "ADDRESS" for s in spans)


# ── Devre dışı bırakılmış entity ─────────────────────────────────────────────

class TestDisabledNEREntity:
    def test_org_disabled_not_detected(self):
        det = NERDetector({"PERSON"}, model="en_core_web_sm")  # ORG disabled
        spans = det.detect("Tim Cook works at Apple.")
        assert not any(s.entity_type == "ORG" for s in spans)

    def test_all_disabled_returns_empty(self):
        det = NERDetector(set(), model="en_core_web_sm")
        spans = det.detect("John Smith at Microsoft in New York.")
        assert spans == []


# ── Bilinmeyen SpaCy etiketi görmezden gelinmeli ─────────────────────────────

class TestUnknownSpacyLabel:
    def test_unmapped_label_ignored(self, ner):
        """CARDINAL, DATE gibi etiketler mapped değil → ignore."""
        spans = ner.detect("She bought 3 items on Monday.")
        types = {s.entity_type for s in spans}
        assert "CARDINAL" not in types
        assert "DATE"     not in types


# ── NER + Regex hibrit (LLMGuard aracılığıyla) ──────────────────────────────

class TestNERPlusRegex:
    def test_ner_person_with_regex_email(self, guard_with_ner):
        text = "John Doe sent an email to john.doe@company.com"
        result = guard_with_ner.scan(text)
        types = {v.entity_type for v in result.violations}
        assert "EMAIL"  in types
        assert "PERSON" in types

    def test_ner_does_not_duplicate_regex_entity(self, guard_with_ner):
        """Regex ve NER aynı span'i yakalarsa overlap çözümü bir tane döndürmeli."""
        text = "Contact Apple at support@apple.com"
        result = guard_with_ner.scan(text)
        emails = [v for v in result.violations if v.entity_type == "EMAIL"]
        assert len(emails) == 1   # çift tespit olmamalı

    def test_ner_person_hashed_in_output(self, guard_with_ner):
        result = guard_with_ner.scan("Please call John Smith at 0532 111 22 33")
        person_violations = [v for v in result.violations if v.entity_type == "PERSON"]
        for v in person_violations:
            assert v.replacement is not None          # hash action
            assert v.replacement.startswith("[PERSON:")

    def test_ner_address_warned_not_hashed(self, guard_with_ner):
        """ADDRESS varsayılan olarak warn; sanitized metinde değişmemeli."""
        from ai_guard.core.models import Action
        result = guard_with_ner.scan("She is from New York and works in Boston.")
        address_violations = [v for v in result.violations if v.entity_type == "ADDRESS"]
        for v in address_violations:
            assert v.action == Action.WARN


# ── Hata yönetimi ────────────────────────────────────────────────────────────

class TestNERErrorHandling:
    def test_invalid_model_raises_on_init(self):
        with pytest.raises(Exception):
            NERDetector({"PERSON"}, model="nonexistent_model_xyz")

    def test_guard_falls_back_gracefully_on_bad_model(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="ai_guard.guard"):
            guard = LLMGuard(use_ner=True, spacy_model="nonexistent_model_xyz")

        assert any("NER" in r.message or "yüklenemedi" in r.message for r in caplog.records)
        # Regex hâlâ çalışmalı
        result = guard.scan("kart: 4111111111111111")
        assert not result.is_clean

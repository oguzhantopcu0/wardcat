"""
Integration tests: realistic LLM prompt scenarios.
NER is kept disabled → runs without SpaCy installed.
"""

import pytest

from ai_guard import AIGuard

SAMPLE_PROMPT = """
Merhaba, ben Ahmet Yılmaz. Şirketimizin sunucu IP'si 10.0.0.42.
Bana fatih.demir@firma.com adresinden veya 0533 987 65 43 numarasından ulaşabilirsin.
Ödeme için IBAN: TR330006100519786457841326 veya
kredi kartı 4532015112830366 kullanabilirsin.
TC kimliğim 10987654202.
""".strip()


@pytest.fixture
def guard():
    return AIGuard(use_ner=False, salt="entegrasyon-tuz")


def test_multiple_entities_detected(guard):
    result = guard.scan(SAMPLE_PROMPT)
    types = {v.entity_type for v in result.violations}
    assert "EMAIL" in types
    assert "PHONE" in types
    assert "IBAN" in types
    assert "CREDIT_CARD" in types
    assert "IP_ADDRESS" in types
    assert "TC_ID" in types


def test_hashed_entities_not_in_sanitized_text(guard):
    # CREDIT_CARD, IBAN, TC_ID are hashed by default
    result = guard.scan(SAMPLE_PROMPT)
    assert "4532015112830366" not in result.sanitized_text
    assert "TR330006100519786457841326" not in result.sanitized_text
    assert "10987654202" not in result.sanitized_text


def test_sanitized_text_has_placeholders(guard):
    result = guard.scan(SAMPLE_PROMPT)
    assert "[CREDIT_CARD:" in result.sanitized_text
    assert "[IBAN:" in result.sanitized_text
    assert "[TC_ID:" in result.sanitized_text


def test_violation_report_structure(guard):
    result = guard.scan(SAMPLE_PROMPT)
    for v in result.violations:
        assert v.entity_type
        assert v.original
        assert v.start >= 0
        assert v.end > v.start
        assert v.action in ("warn", "hash")

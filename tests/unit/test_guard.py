import pytest

from ai_guard import LLMGuard
from ai_guard.core.models import Action


@pytest.fixture
def guard():
    # NER kapalı → SpaCy kurulu olmasa da testler çalışır
    return LLMGuard(use_ner=False)


def test_clean_text_returns_no_violations(guard):
    result = guard.scan("Merhaba, bugün hava çok güzel.")
    assert result.is_clean


def test_email_detected(guard):
    result = guard.scan("Bana user@example.com adresinden ulaşabilirsin.")
    types = [v.entity_type for v in result.violations]
    assert "EMAIL" in types


def test_hash_action_replaces_text(guard):
    guard.configure_entity("EMAIL", enabled=True, action="hash")
    result = guard.scan("Mail: admin@secret.com")
    assert "admin@secret.com" not in result.sanitized_text
    assert "[EMAIL:" in result.sanitized_text


def test_warn_action_preserves_text(guard):
    guard.configure_entity("EMAIL", enabled=True, action="warn")
    result = guard.scan("Mail: admin@secret.com")
    # warn → orijinal metin değişmemeli
    assert "admin@secret.com" in result.sanitized_text
    assert result.violations[0].action == Action.WARN


def test_salt_changes_hash(guard):
    text = "kart: 4111111111111111"
    guard.configure_entity("CREDIT_CARD", enabled=True, action="hash")

    guard.set_salt("tuz-a")
    result_a = guard.scan(text)

    guard.set_salt("tuz-b")
    result_b = guard.scan(text)

    assert result_a.sanitized_text != result_b.sanitized_text


def test_method_chaining():
    guard = (
        LLMGuard(use_ner=False, salt="x")
        .configure_entity("EMAIL",       enabled=True, action="hash")
        .configure_entity("CREDIT_CARD", enabled=True, action="hash")
        .configure_entity("PHONE",       enabled=False)
    )
    result = guard.scan("email: a@b.com kart: 4111111111111111 tel: 0532 123 45 67")
    types = {v.entity_type for v in result.violations}
    assert "EMAIL"       in types
    assert "CREDIT_CARD" in types
    assert "PHONE"       not in types


def test_scan_result_structure(guard):
    result = guard.scan("TC: 12345678901")
    assert hasattr(result, "original_text")
    assert hasattr(result, "sanitized_text")
    assert hasattr(result, "violations")
    assert hasattr(result, "is_clean")

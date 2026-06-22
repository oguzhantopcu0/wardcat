import pytest

from ai_guard import AIGuard
from ai_guard.core.models import Action


@pytest.fixture
def guard():
    # NER disabled → tests run even without SpaCy installed
    return AIGuard(use_ner=False)


def test_clean_text_returns_no_violations(guard):
    result = guard.scan("Merhaba, bugün hava çok güzel.")
    assert result.is_clean


def test_email_detected(guard):
    result = guard.add_entity("EMAIL").scan("Bana user@example.com adresinden ulaşabilirsin.")
    types = [v.entity_type for v in result.violations]
    assert "EMAIL" in types


def test_hash_action_replaces_text(guard):
    guard.add_entity("EMAIL", action="hash")
    result = guard.scan("Mail: admin@secret.com")
    assert "admin@secret.com" not in result.sanitized_text
    assert "[EMAIL:" in result.sanitized_text


def test_warn_action_preserves_text(guard):
    guard.add_entity("EMAIL", action="warn")
    result = guard.scan("Mail: admin@secret.com")
    # warn → original text should not change
    assert "admin@secret.com" in result.sanitized_text
    assert result.violations[0].action == Action.WARN


def test_salt_changes_hash(guard):
    text = "kart: 4111111111111111"
    guard.add_entity("CREDIT_CARD", action="hash")

    guard.set_salt("tuz-a")
    result_a = guard.scan(text)

    guard.set_salt("tuz-b")
    result_b = guard.scan(text)

    assert result_a.sanitized_text != result_b.sanitized_text


def test_method_chaining():
    guard = (
        AIGuard(use_ner=False, salt="x")
        .add_entity("EMAIL", action="hash")
        .add_entity("CREDIT_CARD", action="hash")
        .remove_entity("PHONE")
    )
    result = guard.scan("email: a@b.com kart: 4111111111111111 tel: 0532 123 45 67")
    types = {v.entity_type for v in result.violations}
    assert "EMAIL" in types
    assert "CREDIT_CARD" in types
    assert "PHONE" not in types


def test_scan_result_structure(guard):
    result = guard.scan("TC: 12345678950")
    assert hasattr(result, "original_text")
    assert hasattr(result, "sanitized_text")
    assert hasattr(result, "violations")
    assert hasattr(result, "is_clean")


def test_scan_result_repr(guard):
    result = guard.scan("TC: 12345678950")
    r = repr(result)
    assert "ScanResult" in r
    assert "is_clean" in r


def test_scan_result_redacted_contains_confidence(guard):
    guard.add_entity("TC_ID", action="hash")
    result = guard.scan("TC: 12345678950")
    d = result.redacted()
    assert "violations" in d
    assert len(d["violations"]) > 0
    assert "confidence" in d["violations"][0]
    assert d["violations"][0]["confidence"] == 1.0


def test_scan_result_redacted_no_raw_pii(guard):
    result = guard.scan("TC: 12345678950")
    d = result.redacted()
    # redacted() must not expose the raw PII value
    assert "original_text" not in d
    for v in d["violations"]:
        assert "original" not in v


def test_confidence_is_1_for_regex_detections(guard):
    result = guard.scan("a@b.com")
    assert all(v.confidence == 1.0 for v in result.violations)


def test_redacted_convenience_wrapper(guard):
    """ai_guard.redacted(result) is equivalent to result.redacted()."""
    import ai_guard

    result = guard.scan("a@b.com")
    d = ai_guard.redacted(result)
    assert "violations" in d
    assert "sanitized_text" in d
    assert "original_text" not in d


# ── Redact action ──────────────────────────────────────────────────────────


class TestRedactAction:
    def test_redact_replaces_with_label(self):
        guard = AIGuard(use_ner=False)
        guard.add_entity("EMAIL", action="redact")
        result = guard.scan("Mail: admin@secret.com")
        assert "admin@secret.com" not in result.sanitized_text
        assert "[EMAIL]" in result.sanitized_text

    def test_redact_replacement_has_no_hash(self):
        guard = AIGuard(use_ner=False)
        guard.add_entity("EMAIL", action="redact")
        result = guard.scan("a@b.com")
        v = result.violations[0]
        assert v.replacement == "[EMAIL]"
        assert ":" not in v.replacement  # no hash suffix

    def test_redact_action_enum(self):
        guard = AIGuard(use_ner=False)
        guard.add_entity("EMAIL", action="redact")
        result = guard.scan("a@b.com")
        assert result.violations[0].action == Action.REDACT

    def test_redact_multiple_occurrences(self):
        guard = AIGuard(use_ner=False)
        guard.add_entity("EMAIL", action="redact")
        result = guard.scan("a@b.com and c@d.com")
        assert result.sanitized_text.count("[EMAIL]") == 2

    def test_add_entity_accepts_redact(self):
        guard = AIGuard(use_ner=False)
        returned = guard.add_entity("EMAIL", action="redact")
        assert returned is guard  # method chaining


# ── Mask action ────────────────────────────────────────────────────────────


class TestMaskAction:
    def test_mask_hides_middle_chars(self):
        guard = AIGuard(use_ner=False)
        guard.add_entity("CREDIT_CARD", action="mask")
        result = guard.scan("kart: 4111111111111111")
        assert "4111111111111111" not in result.sanitized_text
        # entity-aware: credit card shows last 4 digits
        assert result.sanitized_text.endswith("1111")
        assert "*" in result.sanitized_text

    def test_mask_uses_stars(self):
        guard = AIGuard(use_ner=False)
        guard.add_entity("EMAIL", action="mask")
        result = guard.scan("a@b.com")
        replacement = result.violations[0].replacement
        assert "*" in replacement

    def test_mask_short_value_all_stars(self):
        """Values shorter than 4 chars → all stars."""
        from ai_guard.core.engine import DetectionEngine
        from ai_guard.detectors.base import BaseDetector, DetectedSpan

        config = {
            "salt": "",
            "entities": {"X": {"enabled": True, "action": "mask"}},
            "allowlist": [],
            "denylist": [],
            "max_text_bytes": 500_000,
        }

        class _FakeDetector(BaseDetector):
            def detect(self, text, candidates=None):
                return [DetectedSpan("X", "ab", 0, 2)]

        engine = DetectionEngine(config, [_FakeDetector()])
        result = engine.scan("ab")
        assert result.violations[0].replacement == "**"

    def test_mask_action_enum(self):
        guard = AIGuard(use_ner=False)
        guard.add_entity("CREDIT_CARD", action="mask")
        result = guard.scan("4111111111111111")
        assert result.violations[0].action == Action.MASK

    def test_add_entity_accepts_mask(self):
        guard = AIGuard(use_ner=False)
        returned = guard.add_entity("EMAIL", action="mask")
        assert returned is guard


# ── Allowlist ──────────────────────────────────────────────────────────────


class TestAllowlist:
    def test_allowlisted_email_not_flagged(self):
        guard = AIGuard(use_ner=False)
        guard.add_entity("EMAIL", action="warn")
        guard.add_allowlist(["no-reply@company.com"])
        result = guard.scan("Contact: no-reply@company.com")
        assert result.is_clean

    def test_non_allowlisted_email_still_flagged(self):
        guard = AIGuard(use_ner=False)
        guard.add_entity("EMAIL", action="warn")
        guard.add_allowlist(["no-reply@company.com"])
        result = guard.scan("Contact: user@other.com")
        assert any(v.entity_type == "EMAIL" for v in result.violations)

    def test_add_allowlist_method_chaining(self):
        guard = (
            AIGuard(use_ner=False)
            .add_entity("EMAIL", action="warn")
            .add_allowlist(["safe@example.com"])
        )
        result = guard.scan("safe@example.com")
        assert result.is_clean

    def test_add_allowlist_idempotent(self):
        guard = AIGuard(use_ner=False)
        guard.add_allowlist(["safe@example.com"])
        guard.add_allowlist(["safe@example.com"])  # duplicate
        assert guard._config["allowlist"].count("safe@example.com") == 1

    def test_add_allowlist_multiple_values(self):
        guard = AIGuard(use_ner=False)
        guard.add_entity("EMAIL", action="warn")
        guard.add_allowlist(["a@b.com", "c@d.com"])
        result = guard.scan("a@b.com and c@d.com")
        assert result.is_clean


# ── Denylist ───────────────────────────────────────────────────────────────


class TestDenylist:
    def test_denylist_value_always_flagged(self):
        guard = AIGuard(use_ner=False)
        guard.add_entity("PERSON", action="warn")
        guard.add_denylist([{"value": "John Smith", "entity_type": "PERSON"}])
        result = guard.scan("Request from John Smith.")
        assert any(v.entity_type == "PERSON" for v in result.violations)

    def test_denylist_hash_action_applied(self):
        guard = AIGuard(use_ner=False)
        guard.add_entity("PERSON", action="hash")
        guard.add_denylist([{"value": "Jane Doe", "entity_type": "PERSON"}])
        result = guard.scan("User: Jane Doe")
        assert "Jane Doe" not in result.sanitized_text
        assert "[PERSON:" in result.sanitized_text

    def test_denylist_method_chaining(self):
        guard = (
            AIGuard(use_ner=False)
            .add_entity("CUSTOM_SECRET", action="warn")
            .add_denylist([{"value": "ProjectX", "entity_type": "CUSTOM_SECRET"}])
        )
        result = guard.scan("deploying ProjectX today")
        assert any(v.entity_type == "CUSTOM_SECRET" for v in result.violations)

    def test_add_denylist_invalid_entry_raises(self):
        guard = AIGuard(use_ner=False)
        with pytest.raises(ValueError, match="'value' or a 'pattern' key"):
            guard.add_denylist([{"entity_type": "PERSON"}])

    def test_denylist_multiple_occurrences_all_flagged(self):
        guard = AIGuard(use_ner=False)
        guard.add_entity("PERSON", action="warn")
        guard.add_denylist([{"value": "John", "entity_type": "PERSON"}])
        result = guard.scan("John and John again")
        john_violations = [v for v in result.violations if v.original == "John"]
        assert len(john_violations) == 2


# ── Denylist regex ─────────────────────────────────────────────────────────


class TestDenylistRegex:
    def test_pattern_denylist_matches(self):
        guard = AIGuard(use_ner=False)
        guard.add_entity("PERSON", action="warn")
        guard.add_denylist([{"pattern": r"\b(CEO|CTO|CFO)\b", "entity_type": "PERSON"}])
        result = guard.scan("Request from CEO John.")
        assert any(v.entity_type == "PERSON" and v.original == "CEO" for v in result.violations)

    def test_pattern_denylist_multiple_matches(self):
        guard = AIGuard(use_ner=False)
        guard.add_entity("PERSON", action="warn")
        guard.add_denylist([{"pattern": r"\b(CEO|CTO)\b", "entity_type": "PERSON"}])
        result = guard.scan("CEO and CTO attended.")
        titles = {v.original for v in result.violations if v.entity_type == "PERSON"}
        assert "CEO" in titles
        assert "CTO" in titles

    def test_pattern_denylist_with_hash_action(self):
        guard = AIGuard(use_ner=False)
        guard.add_entity("CUSTOM_SECRET", action="hash")
        guard.add_denylist([{"pattern": r"\bPROJECT-\d{4}\b", "entity_type": "CUSTOM_SECRET"}])
        result = guard.scan("Working on PROJECT-1234 today.")
        assert "PROJECT-1234" not in result.sanitized_text
        assert "[CUSTOM_SECRET:" in result.sanitized_text

    def test_pattern_denylist_invalid_regex_raises_in_config(self):
        import tempfile

        import yaml

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml.dump({"denylist": [{"pattern": "[invalid", "entity_type": "X"}]}, f)
            cfg_path = f.name
        from ai_guard.config.loader import load_config

        with pytest.raises(ValueError, match="not valid regex"):
            load_config(cfg_path)

    def test_add_denylist_accepts_pattern_entry(self):
        guard = AIGuard(use_ner=False)
        guard.add_entity("PERSON", action="warn")
        guard.add_denylist([{"pattern": r"\bJohn\b", "entity_type": "PERSON"}])
        result = guard.scan("Hello John")
        assert any(v.entity_type == "PERSON" for v in result.violations)

    def test_add_denylist_no_value_or_pattern_raises(self):
        guard = AIGuard(use_ner=False)
        with pytest.raises(ValueError, match="'value' or a 'pattern' key"):
            guard.add_denylist([{"entity_type": "PERSON"}])


# ── Entity-aware mask ──────────────────────────────────────────────────────


class TestEntityAwareMask:
    def test_email_mask_preserves_domain(self):
        guard = AIGuard(use_ner=False)
        guard.add_entity("EMAIL", action="mask")
        result = guard.scan("user@example.com")
        replacement = result.violations[0].replacement
        # domain (@example.com) must be preserved
        assert "@example.com" in replacement
        assert "user" not in replacement

    def test_credit_card_mask_shows_last_4(self):
        guard = AIGuard(use_ner=False)
        guard.add_entity("CREDIT_CARD", action="mask")
        result = guard.scan("4111111111111111")
        replacement = result.violations[0].replacement
        assert replacement.endswith("1111")
        assert "*" in replacement

    def test_ssn_mask_format(self):
        from ai_guard.core.actions import _mask_value

        result = _mask_value("SSN", "123-45-6789")
        assert result == "***-**-6789"

    def test_iban_mask_shows_country_and_last4(self):
        from ai_guard.core.actions import _mask_value

        iban = "TR330006100519786457841326"
        result = _mask_value("IBAN", iban)
        assert result.startswith("TR")
        assert result.endswith("1326")
        assert "*" in result

    def test_tc_id_mask_shows_last3(self):
        from ai_guard.core.actions import _mask_value

        result = _mask_value("TC_ID", "12345678950")
        assert result.endswith("950")
        assert result.startswith("*")

    def test_default_mask_first2_last2(self):
        from ai_guard.core.actions import _mask_value

        result = _mask_value("UUID", "abcdefgh")
        assert result[:2] == "ab"
        assert result[-2:] == "gh"
        assert "*" in result

    def test_mask_very_short_value_all_stars(self):
        from ai_guard.core.actions import _mask_value

        result = _mask_value("EMAIL", "ab")
        # shorter than 4 chars → all stars
        assert set(result) == {"*"}

"""
Adversarial / evasion input tests.

Purpose: Documents the library's behavior against real-world evasion techniques.
Undetectable cases are marked with `xfail` explaining why they are not caught —
these are known limitations, not bugs.
"""

from __future__ import annotations

import pytest

from ai_guard import AIGuard
from tests.conftest import make_legacy_guard


@pytest.fixture
def g():
    return make_legacy_guard(use_ner=False)


# ══════════════════════════════════════════════════════════════════════════
# A01: Separator variations — card number
# ══════════════════════════════════════════════════════════════════════════


class TestCardSeparatorVariants:
    def test_plain_digits(self, g):
        assert "CREDIT_CARD" in {v.entity_type for v in g.scan("4111111111111111").violations}

    def test_single_space_separator(self, g):
        assert "CREDIT_CARD" in {v.entity_type for v in g.scan("4111 1111 1111 1111").violations}

    def test_dash_separator(self, g):
        assert "CREDIT_CARD" in {v.entity_type for v in g.scan("4111-1111-1111-1111").violations}

    def test_mixed_separator(self, g):
        assert "CREDIT_CARD" in {v.entity_type for v in g.scan("4111 1111-1111 1111").violations}

    def test_double_space_separator_detected(self, g):
        """Double-space separator is now detected: _SEP = r'[ \\-\\.]{0,2}' allows 2 chars."""
        result = g.scan("4111  1111  1111  1111")
        assert "CREDIT_CARD" in {v.entity_type for v in result.violations}

    def test_dot_separator_detected(self, g):
        """Dot separator is now detected: _SEP includes '.' in the separator character class."""
        result = g.scan("4111.1111.1111.1111")
        assert "CREDIT_CARD" in {v.entity_type for v in result.violations}


# ══════════════════════════════════════════════════════════════════════════
# A02: Unicode / visual spoofing
# ══════════════════════════════════════════════════════════════════════════


class TestUnicodeAdversarial:
    def test_normal_email_detected(self, g):
        assert "EMAIL" in {v.entity_type for v in g.scan("ali@test.com").violations}

    def test_cyrillic_homoglyph_detected(self, g):
        """Cyrillic 'а' (U+0430) is now detected: Python's \\w matches Unicode word chars."""
        cyrillic_a = "\u0430"  # Cyrillic lowercase а
        email = f"{cyrillic_a}li@test.com"
        result = g.scan(email)
        assert "EMAIL" in {v.entity_type for v in result.violations}

    def test_emoji_in_text_does_not_break_detection(self, g):
        """PII between emojis should still be detected."""
        result = g.scan("🔒 TC: 12345678950 🔒")
        assert "TC_ID" in {v.entity_type for v in result.violations}

    def test_arabic_text_around_pii(self, g):
        """Arabic text should not break PII detection."""
        result = g.scan("مرحبا 4111111111111111 شكرا")
        assert "CREDIT_CARD" in {v.entity_type for v in result.violations}

    def test_cjk_text_around_pii(self, g):
        """PII between CJK characters should still be captured."""
        result = g.scan("你好 a@b.com 谢谢")
        assert "EMAIL" in {v.entity_type for v in result.violations}


# ══════════════════════════════════════════════════════════════════════════
# A03: Noise / embedding in context
# ══════════════════════════════════════════════════════════════════════════


class TestEmbeddedInNoise:
    def test_pii_in_long_paragraph(self, g):
        prefix = "Sevgili müşterimiz, " * 30
        suffix = " lütfen bilgilerinizi güncel tutun." * 20
        text = prefix + "Kartınız: 4111111111111111" + suffix
        result = g.scan(text)
        assert "CREDIT_CARD" in {v.entity_type for v in result.violations}

    def test_pii_after_many_numbers(self, g):
        """Real PII among fake numbers should be correctly selected."""
        text = " ".join([f"ref-{i:04d}" for i in range(50)]) + " TC: 12345678950"
        result = g.scan(text)
        tc_violations = [v for v in result.violations if v.entity_type == "TC_ID"]
        assert len(tc_violations) == 1
        assert tc_violations[0].original == "12345678950"

    def test_pii_split_across_label_and_value(self, g):
        """PII in label–value format should be detected."""
        for fmt in [
            "Email: ali@test.com",
            "E-posta : ali@test.com",
            "EMAIL=ali@test.com",
            '"email":"ali@test.com"',
        ]:
            result = g.scan(fmt)
            assert "EMAIL" in {v.entity_type for v in result.violations}, (
                f"Format not caught: {fmt!r}"
            )


# ══════════════════════════════════════════════════════════════════════════
# A04: Consecutive / adjacent entities
# ══════════════════════════════════════════════════════════════════════════


class TestConsecutiveEntities:
    def test_two_emails_comma_separated(self, g):
        result = g.scan("a@b.com,c@d.com")
        emails = [v for v in result.violations if v.entity_type == "EMAIL"]
        assert len(emails) == 2

    def test_email_immediately_after_phone(self, g):
        """Both should be detected when separated by a separator."""
        result = g.scan("0532 111 22 33 a@b.com")  # separated by space
        types = {v.entity_type for v in result.violations}
        assert "PHONE" in types
        assert "EMAIL" in types

    @pytest.mark.xfail(
        reason="Adjacent phone+email: email regex captures '33a@b.com', "
        "overlaps with phone, overlap resolution keeps the phone (longer span)"
    )
    def test_email_glued_to_phone_evades(self, g):
        """Adjacent phone+email without separator: email evasion vector."""
        result = g.scan("0532 111 22 33a@b.com")
        types = {v.entity_type for v in result.violations}
        assert "PHONE" in types
        assert "EMAIL" in types

    @pytest.mark.xfail(reason="Two IBANs without separator breaks word boundary")
    def test_two_ibans_no_separator(self, g):
        ibans = "TR330006100519786457841326TR330006100519786457841327"
        result = g.scan(ibans)
        iban_violations = [v for v in result.violations if v.entity_type == "IBAN"]
        assert len(iban_violations) == 2


# ══════════════════════════════════════════════════════════════════════════
# A05: Partial / truncated data
# ══════════════════════════════════════════════════════════════════════════


class TestPartialData:
    def test_partial_card_not_detected(self, g):
        result = g.scan("son 4 hane: **1234")
        assert "CREDIT_CARD" not in {v.entity_type for v in result.violations}

    def test_masked_card_not_detected(self, g):
        result = g.scan("kart: 4111 **** **** 1111")
        assert "CREDIT_CARD" not in {v.entity_type for v in result.violations}

    def test_partial_tc_10_digits_not_detected(self, g):
        result = g.scan("numara: 1234567890")  # 10 digits — not TC
        assert "TC_ID" not in {v.entity_type for v in result.violations}

    def test_partial_iban_not_detected(self, g):
        # IBAN minimum 15 characters — below that should not match
        result = g.scan("TR330006100")  # 11 characters — below regex minimum
        assert "IBAN" not in {v.entity_type for v in result.violations}

    def test_15_char_iban_lookalike_rejected(self, g):
        """15-char IBAN lookalike is correctly rejected by mod-97 checksum validation."""
        result = g.scan("TR3300061005197")
        assert "IBAN" not in {v.entity_type for v in result.violations}


# ══════════════════════════════════════════════════════════════════════════
# A06: Case variations
# ══════════════════════════════════════════════════════════════════════════


class TestCaseVariants:
    def test_email_mixed_case(self, g):
        result = g.scan("ALI@TEST.COM")
        assert "EMAIL" in {v.entity_type for v in result.violations}

    def test_iban_lowercase(self, g):
        result = g.scan("tr330006100519786457841326")
        assert "IBAN" in {v.entity_type for v in result.violations}

    def test_iban_mixed_case(self, g):
        result = g.scan("Tr330006100519786457841326")
        assert "IBAN" in {v.entity_type for v in result.violations}


# ══════════════════════════════════════════════════════════════════════════
# A07: Sanitized text integrity — position shift after replacement
# ══════════════════════════════════════════════════════════════════════════


class TestSanitizedIntegrity:
    def test_all_original_originals_extractable_from_original_text(self, g):
        """The original field of each violation should be extractable from original_text."""
        texts = [
            "TC: 12345678950 kart: 4111111111111111 mail: x@y.com",
            "IBAN: TR330006100519786457841326 tel: 0532 111 22 33",
            "ip: 10.0.0.1 posta: 34000 adres: Atatürk Caddesi No:5",
        ]
        for text in texts:
            result = g.scan(text)
            for v in result.violations:
                extracted = text[v.start : v.end]
                assert extracted == v.original, (
                    f"[{v.entity_type}] expected={v.original!r}, "
                    f"extracted from position={extracted!r} pos=[{v.start}:{v.end}]"
                )

    def test_sanitized_text_length_accounts_for_replacements(self, g):
        g2 = AIGuard(use_ner=False)
        g2.add_entity("EMAIL", action="hash")
        text = "prefix a@b.com suffix"
        result = g2.scan(text)
        v = next(v for v in result.violations if v.entity_type == "EMAIL")
        # len(replacement) - len(original) = delta
        expected_delta = len(v.replacement) - len(v.original)
        actual_delta = len(result.sanitized_text) - len(result.original_text)
        assert actual_delta == expected_delta

    def test_non_pii_parts_unchanged_in_sanitized(self, g):
        """Non-sensitive parts of the text should be preserved in sanitized_text."""
        prefix = "BAŞLANGIÇ "
        suffix = " SONUÇ"
        text = prefix + "4111111111111111" + suffix
        result = g.scan(text)
        # CREDIT_CARD default → hash; prefix and suffix should not change
        assert result.sanitized_text.startswith(prefix)
        assert result.sanitized_text.endswith(suffix)

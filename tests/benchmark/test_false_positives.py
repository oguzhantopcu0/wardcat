"""
False positive benchmark suite.

Verifies that legitimate text does NOT trigger PII detection for each entity type.
These tests catch regex patterns that are too broad and produce noise in production.
"""

from __future__ import annotations

from wardcat import Wardcat
from wardcat.detectors.regex_detector import RegexDetector


def _regex(entities):
    return RegexDetector(set(entities))


def _violations(entity, texts):
    """Return total violation count for the given entity across all texts."""
    det = _regex({entity})
    total = 0
    for text in texts:
        total += len(det.detect(text))
    return total


class TestCreditCardFalsePositives:
    NOT_CARDS = [
        "Order #1234567890123",
        "Part number: 9999999999999999",
        "Document ID: 1234 5678 9012",
        "Phone: +90 555 123 45 67",
        "IP: 192.168.1.1",
        "2024-01-15",
        "Version 3.14159265358979",
    ]

    def test_no_false_positives(self):
        assert _violations("CREDIT_CARD", self.NOT_CARDS) == 0


class TestEmailFalsePositives:
    NOT_EMAILS = [
        "user @domain.com",
        "@nodomain",
        "user@",
        "just-text-here",
        "price: $15.99",
    ]

    def test_no_false_positives(self):
        assert _violations("EMAIL", self.NOT_EMAILS) == 0


class TestPhoneFalsePositives:
    NOT_PHONES = [
        "12345",
        "123 456 7890",
        "Year: 2024",
        "Ref: 5551234567",
        "v1.2.3",
    ]

    def test_no_false_positives(self):
        assert _violations("PHONE", self.NOT_PHONES) == 0


class TestPostalCodeFalsePositives:
    NOT_POSTAL = [
        "AB12345",
        "00001",
        "82001",
        "99999",
        "1234",
        "Product-34100",
    ]

    def test_no_false_positives(self):
        assert _violations("POSTAL_CODE", self.NOT_POSTAL) == 0


class TestSSNFalsePositives:
    NOT_SSN = [
        "000-12-3456",
        "666-12-3456",
        "900-12-3456",
        "123-00-3456",
        "123-12-0000",
        "123456789",
        "Date: 2024-01-15",
        "v1-23-4567",
    ]

    def test_no_false_positives(self):
        assert _violations("SSN", self.NOT_SSN) == 0


class TestEUNationalIDFalsePositives:
    NOT_EU_ID = [
        "12345678",  # DNI: 8 digits, no check letter
        "12345678I",  # DNI: I is not a valid check letter
        "1234567890",  # 10 digits, not matching any pattern
        "X1234567",  # NIE: missing check letter
        "012011234567890",  # INSEE: first digit must be 1 or 2
        "113001234567890",  # INSEE: month 00 invalid
        "113131234567890",  # INSEE: month 13 invalid
    ]

    def test_no_false_positives(self):
        assert _violations("EU_NATIONAL_ID", self.NOT_EU_ID) == 0


class TestIBANFalsePositives:
    NOT_IBAN = [
        "TR00000000000000000000000000",
        "GB12ABCD12345678901234",
        "random text 1234567890",
    ]

    def test_no_false_positives(self):
        assert _violations("IBAN", self.NOT_IBAN) == 0


class TestJWTFalsePositives:
    NOT_JWT = [
        "eyJust a word",
        "noteyJ.atall",
        "eyJ",
    ]

    def test_no_false_positives(self):
        assert _violations("JWT", self.NOT_JWT) == 0


class TestUUIDFalsePositives:
    NOT_UUID = [
        "12345678-1234-1234-1234",
        "12345678-1234-1234-1234-12345",
        "not-a-uuid-at-all",
        "ZZZZZZZZ-ZZZZ-ZZZZ-ZZZZ-ZZZZZZZZZZZZ",
    ]

    def test_no_false_positives(self):
        assert _violations("UUID", self.NOT_UUID) == 0


class TestFullPipelineFalsePositives:
    """Run clean prose through the full guard and expect no violations."""

    CLEAN_TEXTS = [
        "The meeting is scheduled for Monday at 3pm.",
        "Please review the attached document.",
        "The total comes to $199.99 including tax.",
        "Version 2.0.1 was released last quarter.",
        "Contact our support team for assistance.",
        "The server responded with HTTP 200 OK.",
        "Today's date is 2024-01-15.",
        "The temperature is 37.5 degrees Celsius.",
    ]

    def test_clean_prose_no_violations(self):
        guard = Wardcat()
        for text in self.CLEAN_TEXTS:
            result = guard.scan(text)
            assert result.is_clean, (
                f"False positive in: {text!r}\n"
                f"Violations: {[(v.entity_type, v.original) for v in result.violations]}"
            )


# ── ADDRESS False Positives ───────────────────────────────────────────────────


class TestAddressFalsePositives:
    """
    Address regex is the broadest pattern — most likely to produce false positives.
    These texts should NOT trigger ADDRESS detection.
    """

    NOT_ADDRESSES = [
        "Please review section 3 of the document.",
        "The team completed 5 tasks this sprint.",
        "Error code: 404",
        "Chapter 2: Introduction",
        "Step 1: Open the terminal.",
        "Revision 10: updated terms.",
        "Track 3 was the most popular song.",
        "Table 5 shows the results.",
    ]

    def test_no_false_positives(self):
        det = _regex({"ADDRESS"})
        for text in self.NOT_ADDRESSES:
            spans = det.detect(text)
            assert not spans, (
                f"False positive ADDRESS in: {text!r}\nMatched: {[s.text for s in spans]}"
            )


# ── scan_batch_workers config ─────────────────────────────────────────────────


class TestScanBatchWorkersConfig:
    def test_default_workers_from_config(self):
        """scan_batch should use config value when max_workers not specified."""
        from wardcat.config.loader import load_config

        cfg = load_config()
        assert cfg["scan_batch_workers"] == 4

    def test_explicit_max_workers_override(self):
        guard = Wardcat().add_entity("EMAIL", "warn")
        texts = ["a@b.com"] * 8
        results = guard.scan_batch(texts, max_workers=2)
        assert len(results) == 8
        assert all(not r.is_clean for r in results)

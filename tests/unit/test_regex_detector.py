import pytest

from ai_guard.detectors.regex_detector import RegexDetector

ALL_ENTITIES = {
    "CREDIT_CARD", "EMAIL", "PHONE", "IBAN", "IP_ADDRESS", "TC_ID",
    "UUID", "SSN", "MAC_ADDRESS", "JWT", "IPv6", "NIN",
}


@pytest.fixture
def detector():
    return RegexDetector(ALL_ENTITIES)


class TestCreditCard:
    def test_visa(self, detector):
        spans = detector.detect("kartım: 4111111111111111 ödeme yaptım")
        types = [s.entity_type for s in spans]
        assert "CREDIT_CARD" in types

    def test_no_false_positive(self, detector):
        spans = detector.detect("sipariş no: 123456")
        assert not any(s.entity_type == "CREDIT_CARD" for s in spans)


class TestEmail:
    def test_standard(self, detector):
        spans = detector.detect("bana ali@example.com adresine yaz")
        assert any(s.entity_type == "EMAIL" and s.text == "ali@example.com" for s in spans)


class TestPhone:
    def test_turkish_format(self, detector):
        spans = detector.detect("telefon: 0532 123 45 67")
        assert any(s.entity_type == "PHONE" for s in spans)


class TestIBAN:
    def test_tr_iban(self, detector):
        spans = detector.detect("IBAN: TR330006100519786457841326")
        assert any(s.entity_type == "IBAN" for s in spans)


class TestIPAddress:
    def test_ipv4(self, detector):
        spans = detector.detect("sunucu: 192.168.1.100")
        assert any(s.entity_type == "IP_ADDRESS" and s.text == "192.168.1.100" for s in spans)


class TestTCID:
    def test_valid_tc(self, detector):
        spans = detector.detect("TC kimlik: 12345678950")
        assert any(s.entity_type == "TC_ID" for s in spans)


class TestUUID:
    def test_standard_uuid(self, detector):
        spans = detector.detect("id: 550e8400-e29b-41d4-a716-446655440000")
        assert any(s.entity_type == "UUID" for s in spans)

    def test_uppercase_uuid(self, detector):
        spans = detector.detect("ID: 550E8400-E29B-41D4-A716-446655440000")
        assert any(s.entity_type == "UUID" for s in spans)

    def test_no_false_positive(self, detector):
        spans = detector.detect("version: 1.2.3")
        assert not any(s.entity_type == "UUID" for s in spans)


class TestSSN:
    def test_valid_ssn(self, detector):
        spans = detector.detect("SSN: 123-45-6789")
        assert any(s.entity_type == "SSN" for s in spans)

    def test_invalid_prefix_000(self, detector):
        spans = detector.detect("000-45-6789")
        assert not any(s.entity_type == "SSN" for s in spans)

    def test_invalid_prefix_666(self, detector):
        spans = detector.detect("666-45-6789")
        assert not any(s.entity_type == "SSN" for s in spans)


class TestMACAddress:
    def test_colon_separated(self, detector):
        spans = detector.detect("mac: 00:1A:2B:3C:4D:5E")
        assert any(s.entity_type == "MAC_ADDRESS" for s in spans)

    def test_dash_separated(self, detector):
        spans = detector.detect("mac: 00-1A-2B-3C-4D-5E")
        assert any(s.entity_type == "MAC_ADDRESS" for s in spans)

    def test_no_false_positive(self, detector):
        spans = detector.detect("ratio: 3:2")
        assert not any(s.entity_type == "MAC_ADDRESS" for s in spans)


class TestJWT:
    def test_valid_jwt(self, detector):
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        spans = detector.detect(f"token: {jwt}")
        assert any(s.entity_type == "JWT" for s in spans)

    def test_no_false_positive(self, detector):
        spans = detector.detect("key: abc.def.ghi")
        assert not any(s.entity_type == "JWT" for s in spans)


class TestIPv6:
    def test_full_ipv6(self, detector):
        spans = detector.detect("addr: 2001:0db8:85a3:0000:0000:8a2e:0370:7334")
        assert any(s.entity_type == "IPv6" for s in spans)

    def test_compressed_ipv6(self, detector):
        spans = detector.detect("addr: 2001:db8::8a2e:0370:7334")
        assert any(s.entity_type == "IPv6" for s in spans)


class TestNIN:
    def test_valid_nin(self, detector):
        spans = detector.detect("NIN: AB123456C")
        assert any(s.entity_type == "NIN" for s in spans)

    def test_no_false_positive(self, detector):
        spans = detector.detect("code: XY99")
        assert not any(s.entity_type == "NIN" for s in spans)


class TestDisabledEntity:
    def test_disabled_entity_not_detected(self):
        detector = RegexDetector({"EMAIL"})   # only EMAIL active
        spans = detector.detect("kart: 4111111111111111 mail: a@b.com")
        types = {s.entity_type for s in spans}
        assert "CREDIT_CARD" not in types
        assert "EMAIL" in types


from ai_guard.detectors.regex_detector import _validate_iban, _validate_tc_id


class TestValidateIBAN:
    def test_short_string_returns_false(self):
        assert _validate_iban("TR") is False

    def test_invalid_checksum_returns_false(self):
        assert _validate_iban("TR000006100519786457841326") is False

    def test_valid_iban_returns_true(self):
        assert _validate_iban("TR330006100519786457841326") is True

    def test_iban_with_spaces_normalized(self):
        assert _validate_iban("TR33 0006 1005 1978 6457 8413 26") is True


class TestValidateTCID:
    def test_too_short_returns_false(self):
        assert _validate_tc_id("1234567890") is False

    def test_starts_with_zero_returns_false(self):
        assert _validate_tc_id("01234567890") is False

    def test_non_digits_returns_false(self):
        assert _validate_tc_id("1234567890A") is False

    def test_wrong_checksum_d9_returns_false(self):
        # Valid format but wrong checksum
        assert _validate_tc_id("12345678900") is False

    def test_valid_tc_returns_true(self):
        assert _validate_tc_id("12345678950") is True


# ── G1 bypass fix tests ────────────────────────────────────────────────────

class TestCreditCardSeparatorFixes:
    """G1a: credit card with double-space and dot separators."""

    def test_double_space_separator(self, detector):
        spans = detector.detect("card: 4111  1111  1111  1111")
        assert any(s.entity_type == "CREDIT_CARD" for s in spans)

    def test_dot_separator(self, detector):
        spans = detector.detect("card: 4111.1111.1111.1111")
        assert any(s.entity_type == "CREDIT_CARD" for s in spans)

    def test_single_space_still_works(self, detector):
        spans = detector.detect("card: 4111 1111 1111 1111")
        assert any(s.entity_type == "CREDIT_CARD" for s in spans)

    def test_dash_separator_still_works(self, detector):
        spans = detector.detect("card: 4111-1111-1111-1111")
        assert any(s.entity_type == "CREDIT_CARD" for s in spans)


class TestConnectionStringCredentials:
    """Passwords embedded in URI userinfo (scheme://user:password@host)."""

    @pytest.fixture
    def detector(self):
        return RegexDetector({"EMAIL", "CUSTOM_SECRET"})

    def test_password_detected_as_secret(self, detector):
        spans = detector.detect(
            "DATABASE_URL=postgresql://admin:Sup3rS3cr3t@db.prod.internal:5432/appdb"
        )
        secret = next(s for s in spans if s.entity_type == "CUSTOM_SECRET")
        assert secret.text == "Sup3rS3cr3t"

    def test_no_spurious_email_for_password_host(self, detector):
        spans = detector.detect("postgresql://admin:Sup3rS3cr3t@db.prod.internal")
        # The "password@host" must NOT be reported as an email.
        assert not any(s.entity_type == "EMAIL" for s in spans)

    def test_empty_user_credential(self, detector):
        spans = detector.detect("redis://:r3d1sP_ss@cache:6379")
        assert any(
            s.entity_type == "CUSTOM_SECRET" and s.text == "r3d1sP_ss" for s in spans
        )

    def test_url_encoded_password(self, detector):
        spans = detector.detect("mongodb+srv://user:p%40ss123@cluster0.mongodb.net")
        assert any(
            s.entity_type == "CUSTOM_SECRET" and s.text == "p%40ss123" for s in spans
        )

    def test_real_email_unaffected(self, detector):
        spans = detector.detect("Contact john@acme.com for details.")
        assert any(s.entity_type == "EMAIL" and s.text == "john@acme.com" for s in spans)

    def test_plain_url_no_credential(self, detector):
        spans = detector.detect("See https://example.com/path for docs.")
        assert not any(s.entity_type == "CUSTOM_SECRET" for s in spans)

    def test_password_offset_correct(self, detector):
        text = "postgresql://admin:Sup3rS3cr3t@db.prod.internal"
        secret = next(
            s for s in detector.detect(text) if s.entity_type == "CUSTOM_SECRET"
        )
        assert text[secret.start:secret.end] == "Sup3rS3cr3t"

    def test_secret_disabled_still_suppresses_email(self):
        # When CUSTOM_SECRET is off, the password is not reported — but the
        # spurious "password@host" email must still be suppressed.
        det = RegexDetector({"EMAIL"})
        spans = det.detect("postgresql://admin:Sup3rS3cr3t@db.prod.internal")
        assert not any(s.entity_type == "EMAIL" for s in spans)


class TestEmailCyrillicHomoglyph:
    """G1b: Cyrillic lookalike characters in email local-part."""

    def test_cyrillic_a_in_local_part(self, detector):
        # Cyrillic 'а' (U+0430) looks like Latin 'a'
        cyrillic_email = "аli@example.com"
        spans = detector.detect(f"email: {cyrillic_email}")
        assert any(s.entity_type == "EMAIL" for s in spans)

    def test_standard_ascii_email_still_works(self, detector):
        spans = detector.detect("email: ali@example.com")
        assert any(s.entity_type == "EMAIL" for s in spans)


class TestCustomPatternsInRegexDetector:
    """G6: custom patterns in RegexDetector."""

    def test_custom_pattern_detected(self):
        custom = {
            "EMPLOYEE_ID": {"pattern": r"\bEMP-\d{6}\b", "action": "hash"},
        }
        det = RegexDetector(set(), custom_patterns=custom)
        spans = det.detect("employee EMP-123456 signed in")
        assert any(s.entity_type == "EMPLOYEE_ID" and s.text == "EMP-123456" for s in spans)

    def test_custom_pattern_with_warn_action(self):
        custom = {
            "PROJECT_CODE": {"pattern": r"\bPRJ-[A-Z]{3}-\d{4}\b", "action": "warn"},
        }
        det = RegexDetector(set(), custom_patterns=custom)
        spans = det.detect("project PRJ-ABC-1234 budget approved")
        assert any(s.entity_type == "PROJECT_CODE" for s in spans)

    def test_custom_and_builtin_patterns_together(self):
        custom = {
            "EMPLOYEE_ID": {"pattern": r"\bEMP-\d{6}\b", "action": "hash"},
        }
        det = RegexDetector({"EMAIL"}, custom_patterns=custom)
        text = "user ali@example.com with id EMP-123456"
        spans = det.detect(text)
        entity_types = {s.entity_type for s in spans}
        assert "EMAIL" in entity_types
        assert "EMPLOYEE_ID" in entity_types

    def test_invalid_custom_pattern_skipped(self):
        """An invalid regex in custom_patterns should be skipped (logged as warning)."""
        custom = {
            "BAD_PATTERN": {"pattern": r"[invalid", "action": "warn"},
        }
        # Should not raise — just skip the bad pattern
        det = RegexDetector(set(), custom_patterns=custom)
        spans = det.detect("test text")
        assert not any(s.entity_type == "BAD_PATTERN" for s in spans)

    def test_safe_finditer_timeout_returns_empty(self, caplog):
        """When a custom pattern times out, _safe_finditer returns [] and logs a warning."""
        import concurrent.futures
        import logging
        import re
        from unittest.mock import MagicMock, patch

        from ai_guard.detectors.regex_detector import _safe_finditer

        pattern = re.compile(r"\w+")
        mock_future = MagicMock()
        mock_future.result.side_effect = concurrent.futures.TimeoutError()
        mock_executor = MagicMock()
        mock_executor.submit.return_value = mock_future
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_executor)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch("ai_guard.detectors.regex_detector.concurrent.futures.ThreadPoolExecutor",
                  return_value=mock_ctx),
            caplog.at_level(logging.WARNING, logger="ai_guard.detectors.regex_detector"),
        ):
            result = _safe_finditer(pattern, "hello world")

        assert result == []
        assert any("timed out" in r.message.lower() for r in caplog.records)


class TestChecksumEdgeCases:
    """Tests that exercise internal validator edge cases for coverage."""

    def test_tc_id_passes_first_check_fails_second(self, detector):
        """Number passes the odd/even sum check but fails the total sum check → not detected."""
        # "10000000079": first check 7==7 passes, second check 8≠9 fails
        spans = detector.detect("10000000079")
        assert not any(s.entity_type == "TC_ID" for s in spans)

    def test_tc_id_invalid_checksum_not_detected(self, detector):
        """Randomly wrong TC_ID checksum → not detected."""
        # Modify last digit of a valid TC_ID
        spans = detector.detect("12345678951")   # valid is 12345678950
        assert not any(s.entity_type == "TC_ID" for s in spans)

    def test_iban_invalid_country_code_not_detected(self, detector):
        """IBAN regex matches but country code is not in SWIFT registry → not detected."""
        # "XX" is not a valid IBAN country code
        fake_iban = "XX330006100519786457841326"
        spans = detector.detect(fake_iban)
        assert not any(s.entity_type == "IBAN" for s in spans)

    def test_iban_wrong_checksum_not_detected(self, detector):
        """IBAN regex matches, valid country, but mod-97 check fails → not detected."""
        # Valid TR IBAN but last digit changed
        spans = detector.detect("TR330006100519786457841327")  # valid ends in 6
        assert not any(s.entity_type == "IBAN" for s in spans)

    def test_validate_iban_too_short(self):
        from ai_guard.detectors.regex_detector import _validate_iban
        assert _validate_iban("TR3") is False

    def test_validate_tc_id_too_short(self):
        from ai_guard.detectors.regex_detector import _validate_tc_id
        assert _validate_tc_id("1234") is False

    def test_validate_tc_id_starts_with_zero(self):
        from ai_guard.detectors.regex_detector import _validate_tc_id
        assert _validate_tc_id("01234567890") is False

    def test_validate_tc_id_non_digit(self):
        from ai_guard.detectors.regex_detector import _validate_tc_id
        assert _validate_tc_id("1234567890a") is False


# ── VEHICLE_PLATE ──────────────────────────────────────────────────────────

class TestVehiclePlate:
    @pytest.fixture
    def detector(self):
        from ai_guard.detectors.regex_detector import RegexDetector
        return RegexDetector({"VEHICLE_PLATE"})

    @pytest.mark.parametrize("plate", [
        "34 ABC 123",
        "06 AZ 1234",
        "81 T 4321",
        "34ABC123",
        "06AZ1234",
        "01 A 12",
        "35 BCD 5678",
    ])
    def test_valid_plates_detected(self, detector, plate):
        spans = detector.detect(f"plaka: {plate}")
        assert any(s.entity_type == "VEHICLE_PLATE" for s in spans), f"not detected: {plate}"

    @pytest.mark.parametrize("text", [
        "00 ABC 123",    # city code 00 — invalid
        "82 ABC 123",    # city code 82 — invalid (only 01–81)
        "AB 123",        # no city code prefix
    ])
    def test_invalid_plates_not_detected(self, detector, text):
        spans = detector.detect(text)
        assert not any(s.entity_type == "VEHICLE_PLATE" for s in spans), f"falsely detected: {text}"

    def test_vehicle_plate_in_guard(self):
        from ai_guard import LLMGuard
        guard = LLMGuard(use_ner=False)
        guard.configure_entity("VEHICLE_PLATE", enabled=True, action="warn")
        result = guard.scan("Araç plakası: 34 ABC 123")
        assert any(v.entity_type == "VEHICLE_PLATE" for v in result.violations)


# ── US_ZIP_CODE ────────────────────────────────────────────────────────────

class TestUSZipCode:
    @pytest.fixture
    def detector(self):
        from ai_guard.detectors.regex_detector import RegexDetector
        return RegexDetector({"US_ZIP_CODE"})

    def test_zip_plus_four(self, detector):
        spans = detector.detect("address ZIP code 90210-1234 here")
        # The full ZIP+4 must be captured — not just the leading 5 digits.
        assert any(
            s.entity_type == "US_ZIP_CODE" and s.text == "90210-1234" for s in spans
        )

    def test_labeled_bare_zip(self, detector):
        spans = detector.detect("ZIP: 90210")
        assert any(s.entity_type == "US_ZIP_CODE" for s in spans)

    def test_plain_five_digits_not_matched(self, detector):
        # Bare 5 digits without label or +4 are too ambiguous to flag.
        spans = detector.detect("order number 90210 shipped")
        assert not any(s.entity_type == "US_ZIP_CODE" for s in spans)


# ── FINANCIAL_AMOUNT ───────────────────────────────────────────────────────

class TestFinancialAmount:
    @pytest.fixture
    def detector(self):
        from ai_guard.detectors.regex_detector import RegexDetector
        return RegexDetector({"FINANCIAL_AMOUNT"})

    @pytest.mark.parametrize("amount", [
        "45.000 TL",
        "350.000 TL",
        "2.1 milyon TL",
        "$85,000",
        "€12.500",
        "₺47.3 milyon",
    ])
    def test_amounts_detected(self, detector, amount):
        spans = detector.detect(f"tutar {amount} olarak")
        assert any(s.entity_type == "FINANCIAL_AMOUNT" for s in spans), amount

    def test_financial_amount_wired_into_guard(self):
        # Regression: the pattern existed but was missing from the guard's
        # _REGEX_ENTITIES set, so enabling it had no effect.
        from ai_guard import LLMGuard
        guard = LLMGuard(use_ner=False)
        guard.configure_entity("FINANCIAL_AMOUNT", enabled=True, action="redact")
        result = guard.scan("Sözleşme bedeli 2.1 milyon TL olarak belirlendi.")
        assert any(v.entity_type == "FINANCIAL_AMOUNT" for v in result.violations)

    def test_vehicle_plate_in_turkish_entities(self):
        from ai_guard.entity_groups import turkish_entities
        assert "VEHICLE_PLATE" in turkish_entities()

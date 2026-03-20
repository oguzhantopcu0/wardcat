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

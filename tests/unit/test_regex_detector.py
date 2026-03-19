import pytest

from ai_guard.detectors.regex_detector import RegexDetector

ALL_ENTITIES = {"CREDIT_CARD", "EMAIL", "PHONE", "IBAN", "IP_ADDRESS", "TC_ID"}


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
        spans = detector.detect("TC kimlik: 12345678901")
        assert any(s.entity_type == "TC_ID" for s in spans)


class TestDisabledEntity:
    def test_disabled_entity_not_detected(self):
        detector = RegexDetector({"EMAIL"})   # sadece EMAIL aktif
        spans = detector.detect("kart: 4111111111111111 mail: a@b.com")
        types = {s.entity_type for s in spans}
        assert "CREDIT_CARD" not in types
        assert "EMAIL" in types

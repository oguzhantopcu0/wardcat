from wardcat.detectors.regex_detector import RegexDetector

ADDR_ENTITIES = {"ADDRESS", "POSTAL_CODE"}


def detector():
    return RegexDetector(ADDR_ENTITIES)


class TestPostalCode:
    def test_valid_istanbul(self):
        spans = detector().detect("posta kodu: 34000 İstanbul")
        types = [s.entity_type for s in spans]
        assert "POSTAL_CODE" in types

    def test_valid_ankara(self):
        spans = detector().detect("Ankara 06100")
        assert any(s.entity_type == "POSTAL_CODE" for s in spans)

    def test_invalid_code_out_of_range(self):
        # 99xxx is invalid in Turkey
        spans = detector().detect("kod: 99000")
        assert not any(s.entity_type == "POSTAL_CODE" for s in spans)


class TestAddressPattern:
    def test_cadde(self):
        spans = detector().detect("Atatürk Caddesi No:5 adresine gelin")
        assert any(s.entity_type == "ADDRESS" for s in spans)

    def test_sokak(self):
        spans = detector().detect("Gül Sokak No:12 dairesi")
        assert any(s.entity_type == "ADDRESS" for s in spans)

    def test_mahalle(self):
        spans = detector().detect("Bağcılar Mahallesi")
        assert any(s.entity_type == "ADDRESS" for s in spans)

    def test_bulvar(self):
        spans = detector().detect("Cumhuriyet Bulvarı 34. sokak")
        assert any(s.entity_type == "ADDRESS" for s in spans)

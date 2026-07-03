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


class TestTurkishAddressBoundaries:
    def test_no_backward_overcapture_across_sentence(self):
        # The street name must not swallow lowercase filler / cross a sentence.
        text = (
            "adresinden iletişime geçilebilir. İkamet adresi "
            "Bağdat Caddesi No:127 Daire:8, Kadıköy/İstanbul"
        )
        addrs = [s.text for s in detector().detect(text) if s.entity_type == "ADDRESS"]
        assert "Bağdat Caddesi No:127 Daire:8" in addrs
        assert not any("iletişime" in a or "geçilebilir" in a or "İkamet" in a for a in addrs)

    def test_captures_no_and_daire_tail(self):
        addrs = [
            s.text
            for s in detector().detect("Moda Caddesi No:42 Kat:3")
            if s.entity_type == "ADDRESS"
        ]
        assert any("No:42" in a and "Kat:3" in a for a in addrs)

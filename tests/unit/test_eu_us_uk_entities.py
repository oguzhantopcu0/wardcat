"""
Unit tests for EU/US/UK entity types.

Scope:
- UK_POSTAL_CODE: British postal codes (SW1A 1AA format)
- US_ZIP_CODE: US postal codes (ZIP+4 and labeled format)
- EU_NATIONAL_ID: Spanish DNI/NIE, French INSEE
- ADDRESS: French, Spanish, Italian, Dutch, German street patterns
"""

from __future__ import annotations

import pytest

from ai_guard.detectors.regex_detector import RegexDetector

ALL_NEW = {"UK_POSTAL_CODE", "US_ZIP_CODE", "EU_NATIONAL_ID", "ADDRESS"}


@pytest.fixture
def detector():
    return RegexDetector(ALL_NEW)


# ─────────────────────────────────────────────────────────────────────────────
# UK_POSTAL_CODE
# ─────────────────────────────────────────────────────────────────────────────


class TestUKPostalCode:
    def test_central_london(self, detector):
        spans = detector.detect("Address: 10 Downing Street, London SW1A 2AA")
        assert any(s.entity_type == "UK_POSTAL_CODE" and "SW1A" in s.text for s in spans)

    def test_ec_format(self, detector):
        spans = detector.detect("Office: EC1A 1BB, London")
        assert any(s.entity_type == "UK_POSTAL_CODE" for s in spans)

    def test_single_letter_area(self, detector):
        spans = detector.detect("Location: M1 1AE, Manchester")
        assert any(s.entity_type == "UK_POSTAL_CODE" for s in spans)

    def test_two_digit_district(self, detector):
        spans = detector.detect("GU21 6TH Surrey")
        assert any(s.entity_type == "UK_POSTAL_CODE" for s in spans)

    def test_without_space(self, detector):
        spans = detector.detect("postcode: W1A1HQ")
        assert any(s.entity_type == "UK_POSTAL_CODE" for s in spans)

    def test_lowercase(self, detector):
        spans = detector.detect("post: sw1a 2aa")
        assert any(s.entity_type == "UK_POSTAL_CODE" for s in spans)

    def test_no_false_positive_plain_number(self, detector):
        spans = detector.detect("ref: 12345")
        assert not any(s.entity_type == "UK_POSTAL_CODE" for s in spans)

    def test_no_false_positive_short_code(self, detector):
        spans = detector.detect("code: AB")
        assert not any(s.entity_type == "UK_POSTAL_CODE" for s in spans)


# ─────────────────────────────────────────────────────────────────────────────
# US_ZIP_CODE
# ─────────────────────────────────────────────────────────────────────────────


class TestUSZipCode:
    def test_zip_plus_four(self, detector):
        spans = detector.detect("ZIP+4: 10001-1234")
        assert any(s.entity_type == "US_ZIP_CODE" and "10001-1234" in s.text for s in spans)

    def test_washington_dc_zip4(self, detector):
        spans = detector.detect("Washington, DC 20500-0001")
        assert any(s.entity_type == "US_ZIP_CODE" for s in spans)

    def test_labeled_zip(self, detector):
        spans = detector.detect("ZIP: 90210")
        assert any(s.entity_type == "US_ZIP_CODE" for s in spans)

    def test_labeled_zip_code(self, detector):
        spans = detector.detect("ZIP code: 10001")
        assert any(s.entity_type == "US_ZIP_CODE" for s in spans)

    def test_labeled_case_insensitive(self, detector):
        spans = detector.detect("zip: 94102")
        assert any(s.entity_type == "US_ZIP_CODE" for s in spans)

    def test_no_false_positive_plain_five_digits(self, detector):
        # Plain 5 digits without context should not be detected
        spans = detector.detect("order #12345 total: 500")
        assert not any(s.entity_type == "US_ZIP_CODE" for s in spans)

    def test_no_false_positive_phone(self, detector):
        spans = detector.detect("call 12345 extension")
        assert not any(s.entity_type == "US_ZIP_CODE" for s in spans)


# ─────────────────────────────────────────────────────────────────────────────
# EU_NATIONAL_ID
# ─────────────────────────────────────────────────────────────────────────────


class TestEUNationalID:
    class TestSpanishDNI:
        def test_valid_dni(self, detector):
            spans = detector.detect("DNI: 12345678Z")
            assert any(s.entity_type == "EU_NATIONAL_ID" and "12345678Z" in s.text for s in spans)

        def test_dni_another_letter(self, detector):
            spans = detector.detect("documento: 87654321T")
            assert any(s.entity_type == "EU_NATIONAL_ID" for s in spans)

        def test_no_false_positive_invalid_letter(self, detector):
            # I, O, U cannot be DNI check letters
            spans = detector.detect("code: 12345678I")
            assert not any(s.entity_type == "EU_NATIONAL_ID" for s in spans)

        def test_no_false_positive_short(self, detector):
            spans = detector.detect("num: 1234567Z")
            assert not any(s.entity_type == "EU_NATIONAL_ID" for s in spans)

    class TestSpanishNIE:
        def test_nie_x_prefix(self, detector):
            spans = detector.detect("NIE: X1234567L")
            assert any(s.entity_type == "EU_NATIONAL_ID" for s in spans)

        def test_nie_y_prefix(self, detector):
            spans = detector.detect("NIE: Y9876543T")
            assert any(s.entity_type == "EU_NATIONAL_ID" for s in spans)

        def test_nie_z_prefix(self, detector):
            spans = detector.detect("NIE: Z0000001R")
            assert any(s.entity_type == "EU_NATIONAL_ID" for s in spans)

    class TestFrenchINSEE:
        def test_male_insee(self, detector):
            # Male: 1 + year(2) + month(01-12) + 9 digits = 15 digits
            spans = detector.detect("INSEE: 180027512345678")
            assert any(s.entity_type == "EU_NATIONAL_ID" for s in spans)

        def test_female_insee(self, detector):
            spans = detector.detect("numéro: 290117512345612")
            assert any(s.entity_type == "EU_NATIONAL_ID" for s in spans)

        def test_no_false_positive_invalid_month(self, detector):
            # Month 13 is invalid — should not match
            spans = detector.detect("num: 180137512345678")
            assert not any(s.entity_type == "EU_NATIONAL_ID" for s in spans)

        def test_no_false_positive_wrong_prefix(self, detector):
            # 15 digits starting with 3 — not INSEE
            spans = detector.detect("ref: 380027512345678")
            assert not any(s.entity_type == "EU_NATIONAL_ID" for s in spans)


# ─────────────────────────────────────────────────────────────────────────────
# ADDRESS — European street patterns
# ─────────────────────────────────────────────────────────────────────────────


class TestEuropeanAddressPatterns:
    class TestFrench:
        def test_rue(self, detector):
            spans = detector.detect("Adresse: Rue de Rivoli 25, Paris")
            assert any(s.entity_type == "ADDRESS" and "Rue" in s.text for s in spans)

        def test_allee(self, detector):
            spans = detector.detect("Allée des Roses 5")
            assert any(s.entity_type == "ADDRESS" for s in spans)

        def test_impasse(self, detector):
            spans = detector.detect("Impasse du Moulin")
            assert any(s.entity_type == "ADDRESS" for s in spans)

        def test_rue_with_article(self, detector):
            spans = detector.detect("Rue de la Paix, Lyon")
            assert any(s.entity_type == "ADDRESS" and "Rue" in s.text for s in spans)

    class TestSpanish:
        def test_calle(self, detector):
            spans = detector.detect("Dirección: Calle Mayor 10, Madrid")
            assert any(s.entity_type == "ADDRESS" and "Calle" in s.text for s in spans)

        def test_avenida(self, detector):
            spans = detector.detect("Avenida de la Constitución 15")
            assert any(s.entity_type == "ADDRESS" for s in spans)

        def test_plaza(self, detector):
            spans = detector.detect("Plaza Mayor 1, Salamanca")
            assert any(s.entity_type == "ADDRESS" for s in spans)

        def test_paseo(self, detector):
            spans = detector.detect("Paseo de Gracia 92, Barcelona")
            assert any(s.entity_type == "ADDRESS" for s in spans)

    class TestItalian:
        def test_piazza(self, detector):
            spans = detector.detect("Indirizzo: Piazza Navona 5, Roma")
            assert any(s.entity_type == "ADDRESS" and "Piazza" in s.text for s in spans)

        def test_corso(self, detector):
            spans = detector.detect("Corso Buenos Aires 10, Milano")
            assert any(s.entity_type == "ADDRESS" for s in spans)

        def test_viale(self, detector):
            spans = detector.detect("Viale Libia 15, Roma")
            assert any(s.entity_type == "ADDRESS" for s in spans)

        def test_vicolo(self, detector):
            spans = detector.detect("Vicolo del Cinque 18, Roma")
            assert any(s.entity_type == "ADDRESS" for s in spans)

    class TestDutch:
        def test_straat(self, detector):
            spans = detector.detect("Adres: Kalverstraat 152, Amsterdam")
            assert any(s.entity_type == "ADDRESS" and "straat" in s.text for s in spans)

        def test_gracht(self, detector):
            spans = detector.detect("Keizersgracht 174, Amsterdam")
            assert any(s.entity_type == "ADDRESS" for s in spans)

        def test_laan(self, detector):
            spans = detector.detect("Prinsenlaan 45, Rotterdam")
            assert any(s.entity_type == "ADDRESS" for s in spans)

    class TestGerman:
        def test_strasse(self, detector):
            spans = detector.detect("Adresse: Hauptstraße 15, Berlin")
            assert any(s.entity_type == "ADDRESS" and "straße" in s.text.lower() for s in spans)

        def test_strasse_ascii(self, detector):
            spans = detector.detect("Musterstrasse 7, Hamburg")
            assert any(s.entity_type == "ADDRESS" for s in spans)

        def test_weg(self, detector):
            spans = detector.detect("Lindenweg 3, München")
            assert any(s.entity_type == "ADDRESS" for s in spans)

        def test_platz(self, detector):
            spans = detector.detect("Alexanderplatz 1, Berlin")
            assert any(s.entity_type == "ADDRESS" for s in spans)

        def test_allee(self, detector):
            spans = detector.detect("Unter den Linden / Lindenallee 5")
            assert any(s.entity_type == "ADDRESS" for s in spans)


# ─────────────────────────────────────────────────────────────────────────────
# PASSPORT
# ─────────────────────────────────────────────────────────────────────────────


class TestPassport:
    @pytest.fixture
    def det(self):
        return RegexDetector({"PASSPORT"})

    def test_english_keyword(self, det):
        spans = det.detect("Passport: AB1234567")
        assert any(s.entity_type == "PASSPORT" for s in spans)

    def test_passport_number_keyword(self, det):
        spans = det.detect("Passport number: A12345678")
        assert any(s.entity_type == "PASSPORT" for s in spans)

    def test_turkish_keyword(self, det):
        spans = det.detect("Pasaport no: A12345678")
        assert any(s.entity_type == "PASSPORT" for s in spans)

    def test_german_keyword(self, det):
        spans = det.detect("Reisepass Nr: C1234567")
        assert any(s.entity_type == "PASSPORT" for s in spans)

    def test_french_keyword(self, det):
        spans = det.detect("Passeport no: AB123456")
        assert any(s.entity_type == "PASSPORT" for s in spans)

    def test_no_false_positive_without_keyword(self, det):
        # Without the passport keyword context, should not match
        spans = det.detect("reference: AB1234567")
        assert not any(s.entity_type == "PASSPORT" for s in spans)

    def test_no_false_positive_short_number(self, det):
        spans = det.detect("Passport: A123")
        assert not any(s.entity_type == "PASSPORT" for s in spans)


# ─────────────────────────────────────────────────────────────────────────────
# CODICE_FISCALE (Italian personal tax code)
# ─────────────────────────────────────────────────────────────────────────────


class TestCodiceFiscale:
    @pytest.fixture
    def det(self):
        return RegexDetector({"CODICE_FISCALE"})

    def test_valid_codice_fiscale(self, det):
        spans = det.detect("Codice fiscale: RSSMRA85T10A562S")
        assert any(s.entity_type == "CODICE_FISCALE" for s in spans)

    def test_another_valid_code(self, det):
        spans = det.detect("CF: BNCSFN80A01H501T")
        assert any(s.entity_type == "CODICE_FISCALE" for s in spans)

    def test_lowercase(self, det):
        spans = det.detect("codice: rssmra85t10a562s")
        assert any(s.entity_type == "CODICE_FISCALE" for s in spans)

    def test_no_false_positive_short(self, det):
        spans = det.detect("code: RSSMRA85T10A56")
        assert not any(s.entity_type == "CODICE_FISCALE" for s in spans)

    def test_no_false_positive_all_letters(self, det):
        spans = det.detect("ABCDEFGHIJKLMNOP")
        assert not any(s.entity_type == "CODICE_FISCALE" for s in spans)

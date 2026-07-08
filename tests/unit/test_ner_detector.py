"""
SpaCy NER detector tests (requires en_core_web_sm and tr_core_news_md).

pytest -m ner          →  run only this file
pytest -m "ner and tr" →  Turkish-only tests
"""

from __future__ import annotations

import logging

import pytest
import spacy.util

from tests.conftest import make_legacy_guard
from wardcat import Wardcat
from wardcat.detectors.ner_detector import NERDetector, _is_valid_person

pytestmark = pytest.mark.ner  # tag: uv run pytest -m ner

_INSTALLED = set(spacy.util.get_installed_models())
needs_tr = pytest.mark.skipif(
    not any(m.startswith("tr_") for m in _INSTALLED),
    reason="No Turkish SpaCy model installed (run: python -m spacy download tr_core_news_md)",
)


@pytest.fixture(scope="module")
def ner():
    """English NER detector — load once per module (SpaCy is heavy)."""
    return NERDetector({"PERSON", "ORG", "ADDRESS"}, model="en_core_web_sm")


@pytest.fixture(scope="module")
def ner_tr():
    """Turkish NER detector — uses the first installed tr_* model."""
    tr_model = next(m for m in sorted(_INSTALLED) if m.startswith("tr_"))
    return NERDetector({"PERSON", "ORG", "ADDRESS"}, model=tr_model)


@pytest.fixture(scope="module")
def guard_with_ner():
    return make_legacy_guard(use_ner=True, spacy_model="en_core_web_sm")


@pytest.fixture(scope="module")
def guard_tr():
    tr_model = next((m for m in sorted(_INSTALLED) if m.startswith("tr_")), "en_core_web_sm")
    return make_legacy_guard(use_ner=True, spacy_model=tr_model)


# ── PERSON detection ─────────────────────────────────────────────────────────


class TestPersonDetection:
    def test_full_name_detected(self, ner):
        spans = ner.detect("John Smith called us today.")
        assert any(s.entity_type == "PERSON" and "John" in s.text for s in spans)

    def test_titled_name_detected(self, ner):
        spans = ner.detect("Dr. Emily Johnson reviewed the case.")
        persons = [s.text for s in spans if s.entity_type == "PERSON"]
        assert any("Emily" in p or "Johnson" in p for p in persons)

    def test_multiple_persons(self, ner):
        spans = ner.detect("Alice met Bob and Charlie at the office.")
        persons = [s for s in spans if s.entity_type == "PERSON"]
        assert len(persons) >= 2

    def test_person_not_in_clean_text(self, ner):
        spans = ner.detect("The weather is nice today.")
        assert not any(s.entity_type == "PERSON" for s in spans)


# ── ORG detection ────────────────────────────────────────────────────────────


class TestOrgDetection:
    def test_known_company(self, ner):
        spans = ner.detect("Microsoft announced new features.")
        assert any(s.entity_type == "ORG" for s in spans)

    def test_person_and_org_together(self, ner):
        spans = ner.detect("Tim Cook is the CEO of Apple.")
        types = {s.entity_type for s in spans}
        assert "PERSON" in types
        assert "ORG" in types


# ── ADDRESS detection (GPE / LOC → ADDRESS) ──────────────────────────────────


class TestAddressDetection:
    def test_city_detected_as_address(self, ner):
        spans = ner.detect("She lives in New York.")
        assert any(s.entity_type == "ADDRESS" for s in spans)

    def test_country_detected_as_address(self, ner):
        spans = ner.detect("The office is located in Germany.")
        assert any(s.entity_type == "ADDRESS" for s in spans)


# ── Disabled entity ───────────────────────────────────────────────────────────


class TestDisabledNEREntity:
    def test_org_disabled_not_detected(self):
        det = NERDetector({"PERSON"}, model="en_core_web_sm")  # ORG disabled
        spans = det.detect("Tim Cook works at Apple.")
        assert not any(s.entity_type == "ORG" for s in spans)

    def test_all_disabled_returns_empty(self):
        det = NERDetector(set(), model="en_core_web_sm")
        spans = det.detect("John Smith at Microsoft in New York.")
        assert spans == []


# ── Unknown SpaCy label should be ignored ────────────────────────────────────


class TestUnknownSpacyLabel:
    def test_unmapped_label_ignored(self, ner):
        """CARDINAL, DATE and similar labels are not mapped → ignore."""
        spans = ner.detect("She bought 3 items on Monday.")
        types = {s.entity_type for s in spans}
        assert "CARDINAL" not in types
        assert "DATE" not in types


# ── NER + Regex hybrid (via Wardcat) ───────────────────────────────────────


class TestNERPlusRegex:
    def test_ner_person_with_regex_email(self, guard_with_ner):
        text = "John Doe sent an email to john.doe@company.com"
        result = guard_with_ner.scan(text)
        types = {v.entity_type for v in result.violations}
        assert "EMAIL" in types
        assert "PERSON" in types

    def test_ner_does_not_duplicate_regex_entity(self, guard_with_ner):
        """If regex and NER capture the same span, overlap resolution should return only one."""
        text = "Contact Apple at support@apple.com"
        result = guard_with_ner.scan(text)
        emails = [v for v in result.violations if v.entity_type == "EMAIL"]
        assert len(emails) == 1  # no duplicate detection

    def test_ner_person_hashed_in_output(self, guard_with_ner):
        result = guard_with_ner.scan("Please call John Smith at 0532 111 22 33")
        person_violations = [v for v in result.violations if v.entity_type == "PERSON"]
        for v in person_violations:
            assert v.replacement is not None  # hash action
            assert v.replacement.startswith("[PERSON:")

    def test_ner_address_warned_not_hashed(self, guard_with_ner):
        """ADDRESS defaults to warn; the sanitized text should be unchanged."""
        from wardcat.core.models import Action

        result = guard_with_ner.scan("She is from New York and works in Boston.")
        address_violations = [v for v in result.violations if v.entity_type == "ADDRESS"]
        for v in address_violations:
            assert v.action == Action.WARN


# ── PERSON false-positive filter (unit) ──────────────────────────────────────


class TestPersonFilter:
    def test_short_token_filtered(self):
        assert _is_valid_person("TC") is False
        assert _is_valid_person("A") is False

    def test_address_keyword_filtered(self):
        assert _is_valid_person("Moda Caddesi No:42") is False
        assert _is_valid_person("Atatürk Bulvarı") is False
        assert _is_valid_person("Main Street") is False

    def test_digit_in_span_filtered(self):
        assert _is_valid_person("TR33") is False
        assert _is_valid_person("No:42") is False

    def test_all_lowercase_filtered(self):
        assert _is_valid_person("adresine veya") is False
        assert _is_valid_person("veya") is False
        assert _is_valid_person("olan kişi") is False

    def test_valid_names_pass(self):
        assert _is_valid_person("Ahmet Yılmaz") is True
        assert _is_valid_person("John Smith") is True
        assert _is_valid_person("Emily") is True

    def test_person_filter_continue_in_detect(self):
        """NERDetector.detect() should skip PERSON entities that fail _is_valid_person."""
        from unittest.mock import MagicMock, patch

        with patch("wardcat.detectors.ner_detector._load_model") as mock_load:
            mock_nlp = MagicMock()
            mock_load.return_value = mock_nlp

            # Fake entity: labeled PERSON but all-lowercase (fails filter → line 107 continue)
            mock_ent = MagicMock()
            mock_ent.label_ = "PERSON"
            mock_ent.text = "adresine veya"
            mock_ent.start_char = 0
            mock_ent.end_char = 13

            mock_doc = MagicMock()
            mock_doc.ents = [mock_ent]
            mock_nlp.return_value = mock_doc

            det = NERDetector({"PERSON"}, model="mock_model")
            spans = det.detect("adresine veya")

        assert len(spans) == 0  # filtered by _is_valid_person


# ── Turkish NER tests ─────────────────────────────────────────────────────────


class TestTurkishNER:
    @needs_tr
    def test_turkish_person_detected(self, ner_tr):
        spans = ner_tr.detect("Ahmet Yılmaz bir yazılım geliştiricisidir.")
        assert any(s.entity_type == "PERSON" for s in spans), (
            "Expected PERSON detection for 'Ahmet Yılmaz'"
        )

    @needs_tr
    def test_turkish_address_not_person(self, ner_tr):
        """'Moda Caddesi No:42' must not appear as PERSON (false positive filter)."""
        spans = ner_tr.detect("Moda Caddesi No:42 adresinde ikamet etmektedir.")
        person_texts = [s.text for s in spans if s.entity_type == "PERSON"]
        assert not any("Caddesi" in t or "No:" in t for t in person_texts), (
            f"Address mis-labeled as PERSON: {person_texts}"
        )

    @needs_tr
    def test_short_token_not_person(self, ner_tr):
        """'TC' abbreviation must not appear as PERSON."""
        spans = ner_tr.detect("TC kimlik numarası 12345678950 olan kişi.")
        person_texts = [s.text for s in spans if s.entity_type == "PERSON"]
        assert "TC" not in person_texts, "'TC' mis-labeled as PERSON"

    @needs_tr
    def test_turkish_model_actually_loaded(self, caplog):
        """Guard must log which SpaCy model is actually loaded (info level)."""
        tr_model = next(m for m in sorted(_INSTALLED) if m.startswith("tr_"))
        with caplog.at_level(logging.INFO, logger="wardcat.guard"):
            # Detection is opt-in: a NER entity must be enabled for the model to load.
            Wardcat().with_ner(spacy_model=tr_model).add_entity("PERSON")
        loaded = [r.message for r in caplog.records if "SpaCy model loaded" in r.message]
        assert loaded, "Expected 'SpaCy model loaded' info log"
        assert any("tr_" in msg for msg in loaded), (
            f"Expected a tr_* model to be logged, got: {loaded}"
        )

    @needs_tr
    def test_turkish_full_scan(self, guard_tr):
        """Turkish PII scan: name, email, phone, credit card, IBAN all detected."""
        text = (
            "Ahmet Yılmaz, ahmet@example.com adresine veya 0532 123 45 67 numarasına ulaşın. "
            "Kredi kartı: 4111 1111 1111 1111. IBAN: TR33 0006 1005 1978 6457 8413 26"
        )
        result = guard_tr.scan(text)
        types = {v.entity_type for v in result.violations}
        assert "PERSON" in types, "PERSON not detected"
        assert "EMAIL" in types, "EMAIL not detected"
        assert "PHONE" in types, "PHONE not detected"
        assert "CREDIT_CARD" in types, "CREDIT_CARD not detected"
        assert "IBAN" in types, "IBAN not detected"


# ── Model fallback transparency ───────────────────────────────────────────────


class TestModelFallback:
    def test_fallback_logged_when_model_not_installed(self, caplog):
        """Requesting a non-installed model must log a WARNING with fallback info."""
        with caplog.at_level(logging.WARNING, logger="wardcat.guard"):
            make_legacy_guard(
                use_ner=True, spacy_model="xx_fake_model_xyz", spacy_auto_download=False
            )
        # Either fallback warning OR "not installed" warning should appear
        assert any(
            "falling back" in r.message.lower() or "not installed" in r.message.lower()
            for r in caplog.records
        )

    def test_correct_model_logged_at_info(self, caplog):
        """When model is installed, INFO log must show the exact model name."""
        with caplog.at_level(logging.INFO, logger="wardcat.guard"):
            make_legacy_guard(use_ner=True, spacy_model="en_core_web_sm").scan("test")
        assert any(
            "en_core_web_sm" in r.message for r in caplog.records if r.levelno == logging.INFO
        )


# ── Gazetteer stopword filter (no model required) ──────────────────────────────


class TestNERStopwordFilter:
    """_is_all_stopwords runs before any model output is accepted."""

    @pytest.mark.parametrize(
        "text",
        [
            "Senior Backend Engineer",  # job title mislabeled as PERSON
            "New hire",  # HR term mislabeled as ADDRESS
            "T.C.",  # abbreviation
            "Müdür",  # Turkish job title
            "Ingénieur",  # French job title
            "Geschäftsführer",  # German job title
        ],
    )
    def test_all_stopwords_rejected(self, text):
        from wardcat.detectors.ner_detector import _is_all_stopwords

        assert _is_all_stopwords(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "John Anderson",
            "Senior Klaus Müller",  # contains a real name → not all stopwords
            "Ayşe Yılmaz",
        ],
    )
    def test_real_names_pass(self, text):
        from wardcat.detectors.ner_detector import _is_all_stopwords

        assert _is_all_stopwords(text) is False


# ── Error handling ────────────────────────────────────────────────────────────


class TestNERErrorHandling:
    def test_invalid_model_raises_on_init(self):
        with pytest.raises(Exception):
            NERDetector({"PERSON"}, model="nonexistent_model_xyz")

    def test_guard_falls_back_gracefully_on_bad_model(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="wardcat.guard"):
            guard = make_legacy_guard(
                use_ner=True, spacy_model="nonexistent_model_xyz", spacy_auto_download=False
            )  # legacy: enables PERSON so the NER detector is actually built

        assert any(
            "NER" in r.message or "could not be loaded" in r.message or "not installed" in r.message
            for r in caplog.records
        )
        # Regex should still work
        result = guard.scan("kart: 4111111111111111")
        assert not result.is_clean

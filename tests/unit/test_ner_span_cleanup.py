"""How NER spans are cut and filtered before they are reported (no model needed)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from wardcat.detectors.ner_detector import (
    NERDetector,
    _best_line,
    _drop_leading_noise,
    _is_all_stopwords,
    _name_fragment,
)


def detector_returning(text: str, *entities: tuple[str, str]) -> NERDetector:
    """An NERDetector whose model labels each given substring of ``text``."""
    ents = []
    for label, value in entities:
        start = text.index(value)
        ents.append(
            SimpleNamespace(text=value, label_=label, start_char=start, end_char=start + len(value))
        )
    detector = NERDetector.__new__(NERDetector)
    detector.nlp = lambda _: SimpleNamespace(ents=ents)
    detector.enabled_entities = {"PERSON", "ORG"}
    return detector


def reported(text: str, *entities: tuple[str, str]) -> list[tuple[str, str]]:
    spans = detector_returning(text, *entities).detect(text)
    for span in spans:
        assert text[span.start : span.end] == span.text
    return [(s.entity_type, s.text) for s in spans]


class TestLineBreaks:
    def test_a_name_stops_at_the_end_of_its_line(self) -> None:
        text = "Anna Josefsen\nAddress: Main St 4"

        assert reported(text, ("PERSON", "Anna Josefsen\nAddress")) == [("PERSON", "Anna Josefsen")]

    def test_offsets_follow_the_cut(self) -> None:
        assert _best_line("Lena Wirth\r\nAddress", 7, 26) == ("Lena Wirth", 7, 17)

    def test_a_single_line_span_is_untouched(self) -> None:
        assert _best_line("Lena Wirth", 0, 10) == ("Lena Wirth", 0, 10)


class TestMultiLineSpans:
    @pytest.mark.parametrize(
        ("span", "kept"),
        [
            ("Renewals Team\nDalton Inc.", "Dalton Inc"),  # the legal form marks the org
            ("Leanne\n\nFrançois A. Bousquet", "François A. Bousquet"),  # more capitals
            ("Social and Environmental Impact\nThe Global Impact Fund", "Global Impact Fund"),
            ("Fritz-Armstrong\nĐoko", "Fritz-Armstrong"),  # a tie goes to the first line
            ("Sam\n\nSam", "Sam"),
        ],
    )
    def test_the_line_that_names_the_entity_is_kept(self, span: str, kept: str) -> None:
        text = f"x {span} y"
        assert reported(text, ("ORG", span)) == [("ORG", kept)]


class TestMarkupAndDelimiters:
    @pytest.mark.parametrize(
        ("span", "kept"),
        [
            ("Carolyn Hill</name", "Carolyn Hill"),
            ("Dawn Perkins|560=726", "Dawn Perkins"),
            ("Hunter L. Barton/1234567890", "Hunter L. Barton"),
            ("BkCode=1290:::ABC Bank", "ABC Bank"),
            ('Marisa S. Cervantes",976,"2289 Compton Common', "Marisa S. Cervantes"),
            ("Acme Corp - FY2023 Financial", "Acme Corp"),
            ("IT Support Team:**", "IT Support Team"),
        ],
    )
    def test_the_name_fragment_survives_the_markup(self, span: str, kept: str) -> None:
        value, start, end = _name_fragment(span, 10, 10 + len(span))
        assert value == kept
        assert ("." * 10 + span)[start:end] == kept

    def test_a_comma_inside_a_name_is_not_a_delimiter(self) -> None:
        span = "Marshall, Hernandez and Simpson"
        assert _name_fragment(span, 0, len(span))[0] == span

    def test_a_span_that_is_all_digits_and_markup_is_left_for_the_filter(self) -> None:
        span = "Groceries4U/12"
        assert _name_fragment(span, 0, len(span))[0] == span
        assert reported("x Groceries4U/12 y", ("PERSON", span)) == []


class TestLeadingWords:
    @pytest.mark.parametrize(
        ("span", "kept"),
        [
            ("The Garmin Orchestra", "Garmin Orchestra"),
            ("the Roadify Transit", "Roadify Transit"),
            ("Dear John Smith", "John Smith"),
            ("Sayın Zeynep Arslan", "Zeynep Arslan"),
            ("dün Garanti Bankası", "Garanti Bankası"),
        ],
    )
    def test_listed_words_are_trimmed_from_the_front(self, span: str, kept: str) -> None:
        text = f"x {span} y"

        assert reported(text, ("ORG", span)) == [("ORG", kept)]

    @pytest.mark.parametrize(
        "span",
        [
            "de la Cruz",  # a particle is part of the name
            "A Smith",  # an initial, not an article
            "An Nguyen",  # a given name
            "Cher",
            "The",  # never trimmed to nothing
            "Raporu Ayşe Demir",  # unlisted: left alone rather than guessed at
        ],
    )
    def test_anything_else_stays(self, span: str) -> None:
        assert _drop_leading_noise(span, 0, len(span))[0] == span


class TestOrganisationFilters:
    @pytest.mark.parametrize("label", ["SSN", "IBAN", "APO AP", "P.O. Box", "Suite", "ATM"])
    def test_field_labels_and_postal_designators_are_not_organisations(self, label: str) -> None:
        assert _is_all_stopwords(label)
        assert reported(f"{label}: 123", ("ORG", label)) == []

    def test_an_organisation_needs_two_letters(self) -> None:
        assert reported("ג€ x", ("ORG", "ג€")) == []

    @pytest.mark.parametrize("street", ["Pollen Crescent", "Koepenicker Str", "Moda Caddesi"])
    def test_a_street_name_is_not_an_organisation(self, street: str) -> None:
        assert reported(f"{street} 12", ("ORG", street)) == []

    def test_a_street_word_inside_a_name_is_fine(self) -> None:
        text = "She reads the Wall Street Journal."

        assert reported(text, ("ORG", "Wall Street Journal")) == [("ORG", "Wall Street Journal")]

"""PHONE numbers named by the word beside them, without phone_regions."""

from __future__ import annotations

import pytest

from wardcat import Action, Entity, Wardcat
from wardcat.detectors.regex_detector import (
    _CUED_PHONE_AFTER,
    _CUED_PHONE_BEFORE,
    CONF_FUZZY,
    CONF_STRUCTURAL,
    RegexDetector,
)
from wardcat.utils.regex_safety import is_catastrophic


def phones(text: str) -> list[str]:
    return [s.text for s in RegexDetector({"PHONE"}).detect(text) if s.entity_type == "PHONE"]


class TestLabelledNumbers:
    @pytest.mark.parametrize(
        ("text", "number"),
        [
            ("Phone: 905-674-3793", "905-674-3793"),
            ("Personal Info:\nPhone:\n60-56-85-91", "60-56-85-91"),
            ("Mobile: 0490 75 40 81", "0490 75 40 81"),
            ("Fax: 345-899-3560x4587", "345-899-3560x4587"),
            ("Desk: +41 (0)96 471 07 95", "+41 (0)96 471 07 95"),
            ("Please call me at 699 956 915.", "699 956 915"),
            ("Can someone call me on 450 0840", "450 0840"),
            ("My phone number is 905-674-3793, thanks", "905-674-3793"),
            ("tel. no: 212 555 1234", "212 555 1234"),
            ("cep telefonu: 532 987 65 43", "532 987 65 43"),
            ("Handy: 0151 2345 6789", "0151 2345 6789"),
        ],
    )
    def test_a_number_after_its_label_is_found(self, text: str, number: str) -> None:
        assert phones(text) == [number]

    @pytest.mark.parametrize(
        ("text", "number"),
        [
            ("781 1704 office", "781 1704"),
            ("07700 063 966-Office", "07700 063 966"),
            ("(08) 8747 6301 fax", "(08) 8747 6301"),
        ],
    )
    def test_a_number_before_a_signature_label_is_found(self, text: str, number: str) -> None:
        assert phones(text) == [number]


class TestWhatIsNotAPhone:
    @pytest.mark.parametrize(
        "text",
        [
            # No label: a bare digit run is still refused, as before.
            "905-674-3793 and 0490 75 40 81",
            "order number 12345678",
            "Ticket no. 4471 5520 opened",
            "sipariş numarası 4471",
            "hotel 1234567 rooms",
            # A label followed by something that is not a phone number.
            "Phone: 2015-12-22 04:34:22",
            "Phone: 12.03.1988",
            "Phone: 1234",
            "Office: 1200 Main Street",
            "mobile: 4111 1111 1111 1111",
            # "MOB-" is an identifier prefix as often as it is a label.
            "Transaction ID: MOB-123456789",
        ],
    )
    def test_nothing_is_reported(self, text: str) -> None:
        assert phones(text) == []

    def test_a_number_that_only_looks_like_a_date_in_part_is_kept(self) -> None:
        """Danish numbers group in pairs; "28-64-66" is not a day, month and year."""
        assert phones("Phone: 28-64-66-98") == ["28-64-66-98"]


class TestConfidence:
    def test_a_labelled_number_is_fuzzy(self) -> None:
        [span] = RegexDetector({"PHONE"}).detect("Phone: 905-674-3793")

        assert span.confidence == CONF_FUZZY

    def test_the_structural_pattern_keeps_its_tier_and_is_not_duplicated(self) -> None:
        spans = RegexDetector({"PHONE"}).detect("telefon: 0532 123 45 67")

        assert [(s.text, s.confidence) for s in spans] == [("0532 123 45 67", CONF_STRUCTURAL)]

    def test_off_when_phone_is_not_enabled(self) -> None:
        assert RegexDetector({"EMAIL"}).detect("Phone: 905-674-3793") == []


class TestThroughTheGuard:
    def test_masked_by_default_configuration(self) -> None:
        guard = Wardcat(salt="s").add_entity(Entity.PHONE, Action.REDACT)

        result = guard.scan("Name: Anna\nPhone: 0490 75 40 81\nThanks")

        assert "0490 75 40 81" not in result.sanitized_text
        assert result.warnings == []

    def test_with_regions_a_number_is_reported_once(self) -> None:
        pytest.importorskip("phonenumbers")
        guard = Wardcat(salt="s").add_entity(Entity.PHONE, Action.REDACT).with_phone_regions("US")

        result = guard.scan("Phone: 905-674-3793")

        assert len(result.violations) == 1


@pytest.mark.parametrize("pattern", [_CUED_PHONE_BEFORE, _CUED_PHONE_AFTER])
def test_the_patterns_pass_the_redos_screen(pattern) -> None:
    assert not is_catastrophic(pattern)

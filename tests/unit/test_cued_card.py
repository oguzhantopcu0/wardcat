"""CREDIT_CARD numbers named by the word beside them, outside the known issuer ranges.

The built-in pattern offers a digit run to the Luhn check only when its prefix
belongs to a scheme it lists, since about one bare run in ten passes Luhn by
chance. A number from a range it does not list — UnionPay's 62, Mir's 2200-2204,
a private-label or newly issued BIN — was missed even when the text called it a
card. Named as a card and passing Luhn, it is one.
"""

from __future__ import annotations

import pytest

from wardcat import Action, Entity, Wardcat
from wardcat.detectors.regex_detector import (
    _CUED_CARD,
    CONF_CHECKSUM,
    CONF_FUZZY,
    RegexDetector,
)
from wardcat.utils.regex_safety import is_catastrophic

# Synthetic, Luhn-valid, outside every prefix the built-in pattern lists.
UNLISTED_16 = "3512104332181960"
UNIONPAY_16 = "6280654235116151"
SHORT_12 = "584563794026"
LONG_19 = "8600594078161849598"
AMEX_LENGTH_15 = "701231034131642"


def cards(text: str) -> list[str]:
    return [
        s.text
        for s in RegexDetector({"CREDIT_CARD"}).detect(text)
        if s.entity_type == "CREDIT_CARD"
    ]


class TestNamedCards:
    @pytest.mark.parametrize(
        ("text", "number"),
        [
            (f"Credit card {UNLISTED_16} was declined", UNLISTED_16),
            ("card number: 3597 0013 3890 8388", "3597 0013 3890 8388"),
            (f"What is the limit for card {SHORT_12}?", SHORT_12),
            ("my debit card is 6280-6542-3511-6151", "6280-6542-3511-6151"),
            (f"cc {AMEX_LENGTH_15} expires 09/27", AMEX_LENGTH_15),
            ("Kredi kartı numaram 8600 5940 7816 1849 598", "8600 5940 7816 1849 598"),
            (f"Kartımın numarası {UNIONPAY_16}", UNIONPAY_16),
            (f"Kartennummer: {UNLISTED_16}", UNLISTED_16),
            (f"numéro de carte {UNIONPAY_16}", UNIONPAY_16),
            (f"número de tarjeta {LONG_19}", LONG_19),
        ],
    )
    def test_a_luhn_valid_number_after_a_card_word_is_found(self, text: str, number: str) -> None:
        assert cards(text) == [number]


class TestWhatIsNotACard:
    @pytest.mark.parametrize(
        "text",
        [
            # Fails Luhn: the word does not make a wrong number right.
            "card 3512104332181961",
            # No card word: a bare run outside the listed ranges is still refused.
            f"reference {UNLISTED_16}",
            f"order {UNIONPAY_16} shipped",
            # A SIM's ICCID is a Luhn-checked 19-digit number called a card.
            "SIM card 8991752553419283270",
            "sim-card 8991752553419283270",
            # No issuer number starts with 0.
            "card 060464835036",
            # Too short to be a card number.
            "card ending in 4421",
            "card 12345678901",
            # Part of a longer digit run: which digits would be the card is a guess.
            f"card {UNLISTED_16} 1234",
            f"card {UNLISTED_16}1234",
        ],
    )
    def test_nothing_is_reported(self, text: str) -> None:
        assert cards(text) == []


class TestConfidence:
    def test_a_named_card_is_fuzzy(self) -> None:
        [span] = RegexDetector({"CREDIT_CARD"}).detect(f"card {UNLISTED_16}")

        assert span.confidence == CONF_FUZZY

    def test_a_listed_scheme_keeps_its_tier_and_is_not_duplicated(self) -> None:
        spans = RegexDetector({"CREDIT_CARD"}).detect("card 4111 1111 1111 1111")

        assert [(s.text, s.confidence) for s in spans] == [("4111 1111 1111 1111", CONF_CHECKSUM)]

    def test_off_when_credit_card_is_not_enabled(self) -> None:
        assert RegexDetector({"PHONE"}).detect(f"card {UNLISTED_16}") == []


class TestThroughTheGuard:
    def test_redacted(self) -> None:
        guard = Wardcat(salt="s").add_entity(Entity.CREDIT_CARD, Action.REDACT)

        result = guard.scan(f"Please block my credit card {UNIONPAY_16} today")

        assert result.sanitized_text == "Please block my credit card [CREDIT_CARD] today"
        assert result.warnings == []

    def test_invisible_characters_do_not_hide_it(self) -> None:
        guard = Wardcat(salt="s").add_entity(Entity.CREDIT_CARD, Action.REDACT)

        result = guard.scan("card 3512​1043 3218 1960")

        assert result.sanitized_text == "card [CREDIT_CARD]"


def test_the_pattern_passes_the_redos_screen() -> None:
    assert not is_catastrophic(_CUED_CARD)

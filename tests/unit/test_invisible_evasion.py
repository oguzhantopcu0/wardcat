"""Invisible and formatting characters must not hide PII from the regex layer.

A zero-width space inside a card number, an NBSP between its groups or a
fullwidth ``＠`` in an email used to make the value invisible to every
pattern: ``fold_confusables`` preserves length, so it cannot delete a
character. Matching now runs on a prepared copy (``prepare_for_matching``)
and spans are mapped back to the original text, so the whole obfuscated value
— the invisible characters included — is what gets replaced.
"""

from __future__ import annotations

import pytest

from wardcat import Action, Entity, Wardcat
from wardcat.detectors.regex_detector import RegexDetector
from wardcat.utils.normalize import (
    fold_confusables,
    needs_matching_prep,
    prepare_for_matching,
)

# Format (Cf) characters a copy-paste or an attacker can slip into a value.
FORMAT_CHARS = {
    "ZWSP": "​",
    "ZWNJ": "‌",
    "ZWJ": "‍",
    "WORD_JOINER": "⁠",
    "BOM": "﻿",
    "SOFT_HYPHEN": "­",
    "LRM": "‎",
    "RLM": "‏",
    "RLO": "‮",
    "LRI": "⁦",
    "PDI": "⁩",
}

# Space separators (Zs) that are not U+0020: Word and PDF copies produce these.
SPACE_CHARS = {
    "NBSP": " ",
    "NNBSP": " ",
    "THIN": " ",
    "IDEOGRAPHIC": "　",
}

# (entity, clean value, a prefix that cues the type)
VALUES = [
    (Entity.CREDIT_CARD, "4111111111111111", "kart "),
    (Entity.EMAIL, "ali@example.com", "mail "),
    (Entity.TC_ID, "10000000146", "TC "),
    (Entity.IBAN, "TR330006100519786457841326", "IBAN: "),
    (Entity.SSN, "123-45-6789", "SSN: "),
    (Entity.PHONE, "+90 532 123 45 67", "tel "),
]

ENTITIES = [entity for entity, _, _ in VALUES]


def _guard() -> Wardcat:
    guard = Wardcat(salt="s")
    for entity in ENTITIES:
        guard.add_entity(entity, Action.TOKENIZE)
    return guard


def _assert_fully_replaced(guard: Wardcat, prefix: str, obfuscated: str, entity: Entity) -> None:
    text = f"{prefix}{obfuscated} tamam"
    result = guard.scan(text)
    found = [v for v in result.violations if v.entity_type == entity.value]
    assert found, f"{entity.value} hidden by obfuscation: {obfuscated!r}"
    assert any(v.original == obfuscated for v in found), [v.original for v in found]
    # Nothing of the value, the invisible characters included, is left behind.
    assert obfuscated not in result.sanitized_text
    for ch in set(FORMAT_CHARS.values()) | set(SPACE_CHARS.values()):
        if ch in obfuscated:
            assert ch not in result.sanitized_text
    assert result.restore(result.sanitized_text).text == text


@pytest.mark.parametrize("char_name", sorted(FORMAT_CHARS))
@pytest.mark.parametrize(("entity", "value", "prefix"), VALUES, ids=[e.value for e in ENTITIES])
def test_a_format_character_inside_the_value_does_not_hide_it(
    entity: Entity, value: str, prefix: str, char_name: str
) -> None:
    middle = len(value) // 2
    obfuscated = value[:middle] + FORMAT_CHARS[char_name] + value[middle:]
    _assert_fully_replaced(_guard(), prefix, obfuscated, entity)


@pytest.mark.parametrize("char_name", sorted(FORMAT_CHARS))
def test_a_format_character_between_every_digit_does_not_hide_a_card(char_name: str) -> None:
    obfuscated = FORMAT_CHARS[char_name].join("4111111111111111")
    _assert_fully_replaced(_guard(), "kart ", obfuscated, Entity.CREDIT_CARD)


@pytest.mark.parametrize("char_name", sorted(SPACE_CHARS))
@pytest.mark.parametrize(
    ("entity", "grouped"),
    [
        (Entity.CREDIT_CARD, "4111 1111 1111 1111"),
        (Entity.IBAN, "TR33 0006 1005 1978 6457 8413 26"),
        (Entity.PHONE, "+90 532 123 45 67"),
    ],
    ids=["CREDIT_CARD", "IBAN", "PHONE"],
)
def test_a_non_ascii_space_between_groups_does_not_hide_the_value(
    entity: Entity, grouped: str, char_name: str
) -> None:
    obfuscated = grouped.replace(" ", SPACE_CHARS[char_name])
    _assert_fully_replaced(_guard(), "no ", obfuscated, entity)


def test_a_fullwidth_at_sign_does_not_hide_an_email() -> None:
    _assert_fully_replaced(_guard(), "mail ", "ali＠example.com", Entity.EMAIL)


@pytest.mark.parametrize(
    ("script", "zero"),
    [("devanagari", 0x0966), ("bengali", 0x09E6), ("math-bold", 0x1D7CE), ("thai", 0x0E50)],
)
def test_foreign_decimal_digits_do_not_hide_a_card_or_tc_id(script: str, zero: int) -> None:
    def to_script(digits: str) -> str:
        return "".join(chr(zero + int(d)) for d in digits)

    _assert_fully_replaced(_guard(), "kart ", to_script("4111111111111111"), Entity.CREDIT_CARD)
    _assert_fully_replaced(_guard(), "TC ", to_script("10000000146"), Entity.TC_ID)


@pytest.mark.parametrize("char_name", ["ZWSP", "SOFT_HYPHEN", "WORD_JOINER", "RLO"])
def test_a_format_character_does_not_hide_a_denylist_value(char_name: str) -> None:
    guard = (
        Wardcat(salt="s")
        .add_entity(Entity.CUSTOM_SECRET, Action.TOKENIZE)
        .add_denylist(
            [
                {"value": "Proje Zeus", "entity_type": "CUSTOM_SECRET"},
                {"pattern": r"\bPRJ-\d{4}\b", "entity_type": "CUSTOM_SECRET"},
            ]
        )
    )
    ch = FORMAT_CHARS[char_name]
    for obfuscated in (f"Proje Ze{ch}us", f"PRJ-12{ch}34"):
        text = f"kod adı {obfuscated} bitti"
        result = guard.scan(text)
        assert [v.original for v in result.violations] == [obfuscated]
        assert ch not in result.sanitized_text
        assert result.restore(result.sanitized_text).text == text


@pytest.mark.parametrize(
    "text",
    [
        "Aile fotoğrafı 👨‍👩‍👧 çok güzeldi.",
        "می‌خواهم فردا بیایم.",  # Persian ZWNJ is part of the word
        "Toplantı saat üçte, oda 12.",
        "Fiyat ٣ adet için yazıldı.",
    ],
)
def test_clean_text_with_these_characters_stays_clean(text: str) -> None:
    result = _guard().scan(text)
    assert result.violations == []
    assert result.sanitized_text == text


def test_the_option_can_be_switched_off() -> None:
    detector = RegexDetector({"CREDIT_CARD"}, normalize_invisible=False)
    assert detector.detect("kart 4111​1111​1111​1111") == []
    detector = RegexDetector({"CREDIT_CARD"})
    assert len(detector.detect("kart 4111​1111​1111​1111")) == 1


def test_it_is_a_known_config_key() -> None:
    from wardcat.config.loader import _KNOWN_CONFIG_KEYS, load_config

    assert "normalize_invisible" in _KNOWN_CONFIG_KEYS
    assert load_config(None)["normalize_invisible"] is True


class TestPrepareForMatching:
    def test_ascii_needs_nothing(self) -> None:
        assert needs_matching_prep("plain ascii 4111") is False
        assert needs_matching_prep("Türkçe metin, şöyle böyle") is False

    def test_each_kind_is_noticed(self) -> None:
        assert needs_matching_prep("a​b")
        assert needs_matching_prep("a b")
        assert needs_matching_prep("१")

    def test_index_map_points_into_the_original(self) -> None:
        text = "a​b c१d"
        prepared, index_map = prepare_for_matching(text)
        assert prepared == "ab c1d"
        assert [text[i] for i in index_map] == ["a", "b", " ", "c", "१", "d"]

    def test_variation_selectors_are_dropped(self) -> None:
        prepared, _ = prepare_for_matching("4️1\U000e0101")
        assert prepared == "41"

    def test_fullwidth_at_folds(self) -> None:
        assert fold_confusables("ali＠example.com") == "ali@example.com"

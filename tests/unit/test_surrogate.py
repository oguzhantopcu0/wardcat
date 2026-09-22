"""The surrogate action: realistic, shape-preserving, deterministic, reversible."""

from __future__ import annotations

import re

import pytest

from wardcat import Action, ConfigError, Entity, Wardcat
from wardcat.detectors.regex_detector import _validate_iban, _validate_luhn, _validate_tc_id
from wardcat.surrogates import SurrogateAllocator, supports
from wardcat.surrogates._pools import FIRST_NAMES, LAST_NAMES

TEXT = (
    "Ahmet Yılmaz (ahmet.yilmaz@firma.com, 0532 123 45 67) paid with 4111 1111 1111 1111 "
    "to IBAN TR33 0006 1005 1978 6457 8413 26, TC 10000000146, SSN 536-90-4172, ip 10.0.0.7"
)
ENTITIES = [
    Entity.PERSON,
    Entity.EMAIL,
    Entity.PHONE,
    Entity.CREDIT_CARD,
    Entity.IBAN,
    Entity.TC_ID,
    Entity.SSN,
    Entity.IP_ADDRESS,
]


def guard(locale: str = "en", salt: str = "s") -> Wardcat:
    return Wardcat(salt=salt).add_entities(ENTITIES, action=Action.SURROGATE).with_locale(locale)


def by_type(result) -> dict[str, tuple[str, str]]:
    return {v.entity_type: (v.original, v.replacement or "") for v in result.violations}


class TestShapes:
    def test_every_regex_type_keeps_its_shape_and_passes_its_checksum(self) -> None:
        got = by_type(guard("tr").scan(TEXT))
        assert set(got) >= {"EMAIL", "PHONE", "CREDIT_CARD", "IBAN", "TC_ID", "SSN", "IP_ADDRESS"}
        for entity, (original, surrogate) in got.items():
            assert surrogate != original, entity
        assert re.fullmatch(r"[a-z0-9.]+@example\.(com|net|org)", got["EMAIL"][1])
        assert got["PHONE"][1].startswith("0532 ") and len(got["PHONE"][1]) == len(got["PHONE"][0])
        assert _validate_luhn(got["CREDIT_CARD"][1]) and got["CREDIT_CARD"][1].count(" ") == 3
        assert _validate_iban(got["IBAN"][1]) and got["IBAN"][1].startswith("TR")
        assert _validate_tc_id(got["TC_ID"][1])
        assert re.fullmatch(r"\d{3}-\d\d-\d{4}", got["SSN"][1])
        assert got["IP_ADDRESS"][1].startswith(("203.0.113.", "198.51.100.", "192.0.2."))

    @pytest.mark.parametrize("locale", ["en", "tr", "de", "fr"])
    def test_names_come_from_the_locale(self, locale: str) -> None:
        alloc = SurrogateAllocator("s", locale)
        first, last = alloc.surrogate_for("PERSON", "Ada Lovelace").split()
        assert first in FIRST_NAMES[locale] and last in LAST_NAMES[locale]

    def test_case_follows_the_original(self) -> None:
        alloc = SurrogateAllocator("s", "en")
        assert alloc.surrogate_for("PERSON", "ADA LOVELACE").isupper()
        assert alloc.surrogate_for("PERSON", "ada lovelace").islower()
        assert alloc.surrogate_for("PERSON", "Ada").count(" ") == 0

    def test_north_american_numbers_get_the_555_fiction(self) -> None:
        surrogate = SurrogateAllocator("s", "en").surrogate_for("PHONE", "415-555-0142")
        assert re.fullmatch(r"\d{3}-555-01\d\d", surrogate)


class TestDeterminism:
    def test_same_salt_same_surrogate_across_scans(self) -> None:
        a = by_type(guard().scan(TEXT))
        b = by_type(guard().scan("Again: " + TEXT))
        assert a == b

    def test_a_different_salt_gives_unrelated_surrogates(self) -> None:
        a = by_type(guard(salt="one").scan(TEXT))
        b = by_type(guard(salt="two").scan(TEXT))
        assert a["EMAIL"] != b["EMAIL"] or a["CREDIT_CARD"] != b["CREDIT_CARD"]

    def test_two_values_never_share_a_surrogate_in_one_scan(self) -> None:
        alloc = SurrogateAllocator("s", "en")
        seen = {alloc.surrogate_for("PERSON", f"Person Number{i}") for i in range(300)}
        assert len(seen) == 300

    def test_the_same_value_keeps_one_surrogate_within_a_scan(self) -> None:
        result = guard().scan("Ada Lovelace met Ada Lovelace; mail ada@x.io and ada@x.io")
        assert len({v.replacement for v in result.violations if v.entity_type == "EMAIL"}) == 1


class TestRoundTrip:
    def test_restore_puts_every_value_back(self) -> None:
        result = guard("tr").scan(TEXT)
        restored = result.restore(result.sanitized_text)
        assert restored.is_complete
        assert restored.text == TEXT

    def test_reapply_to_surrogate_uses_the_guard_locale(self) -> None:
        result = guard("de").add_entities(ENTITIES, action=Action.REDACT).scan(TEXT)
        derived = result.reapply(Action.SURROGATE)
        for v in derived.violations:
            assert v.action == "surrogate" or v.action == "tokenize"
        assert derived.restore(derived.sanitized_text).is_complete

    def test_a_rescan_finds_the_surrogates_under_the_same_types(self) -> None:
        detector = Wardcat(salt="s").add_entities(ENTITIES, action=Action.REDACT)
        first = guard("tr").scan(TEXT)
        second = detector.scan(first.sanitized_text)
        wanted = {v.entity_type for v in first.violations} - {"PERSON"}  # needs NER
        assert wanted <= {v.entity_type for v in second.violations}


class TestFallback:
    def test_a_type_without_a_generator_is_tokenized_and_says_so(self) -> None:
        assert not supports("DATE_OF_BIRTH")
        result = (
            Wardcat(salt="s")
            .add_entity(Entity.DATE_OF_BIRTH, Action.SURROGATE)
            .scan("doğum tarihi 12.05.1990")
        )
        [v] = result.violations
        assert v.action == "tokenize"
        assert v.replacement.startswith("[DATE_OF_BIRTH_1")

    def test_an_unknown_locale_is_refused(self) -> None:
        with pytest.raises(ConfigError):
            Wardcat(salt="s").with_locale("xx")
        with pytest.raises(ValueError):
            SurrogateAllocator("s", "xx")

    def test_yaml_locale(self, tmp_path) -> None:
        cfg = tmp_path / "p.yaml"
        cfg.write_text("locale: tr\nentities:\n  PERSON: {enabled: true, action: surrogate}\n")
        assert Wardcat(config_path=str(cfg), salt="s")._config["locale"] == "tr"
        cfg.write_text("locale: xx\n")
        with pytest.raises(ConfigError):
            Wardcat(config_path=str(cfg), salt="s")

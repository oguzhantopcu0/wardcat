"""libphonenumber-backed PHONE detection, opt-in via ``with_phone_regions``."""

from __future__ import annotations

import logging

import pytest

from wardcat import Action, Entity, Wardcat
from wardcat.detectors.regex_detector import CONF_FUZZY, CONF_STRUCTURAL, RegexDetector

phonenumbers = pytest.importorskip("phonenumbers", reason="needs the [phone] extra")

# National formats the built-in pattern does not reach, one per numbering plan.
NATIONAL = {
    "US": "905-674-3793",
    "GB": "07700 063 966",
    "BE": "0490 75 40 81",
    "ES": "699 956 915",
}
ALL_REGIONS = list(NATIONAL)


class TestOptIn:
    def test_the_built_in_pattern_is_unchanged_by_default(self) -> None:
        """A base install must behave exactly as it did before this existed."""
        detector = RegexDetector({"PHONE"})

        found = [s.text for s in detector.detect(" and ".join(NATIONAL.values()))]

        assert found == []

    def test_each_national_format_is_found_for_its_own_region(self) -> None:
        for region, number in NATIONAL.items():
            detector = RegexDetector({"PHONE"}, phone_regions=[region])

            found = [s.text for s in detector.detect(f"call {number} today")]

            assert found == [number], region

    def test_e164_still_works_alongside(self) -> None:
        detector = RegexDetector({"PHONE"}, phone_regions=ALL_REGIONS)

        found = [s.text for s in detector.detect("ring +90 532 123 45 67 please")]

        assert found == ["+90 532 123 45 67"]

    def test_regions_are_merged_without_duplicating_a_span(self) -> None:
        """A number valid in two regions must not be reported twice."""
        detector = RegexDetector({"PHONE"}, phone_regions=["US", "CA", "GB"])

        found = detector.detect("call +1 415 555 0142 now")

        assert len(found) == 1


class TestConfidence:
    def test_library_matches_are_fuzzy_not_structural(self) -> None:
        """A numbering-plan check is weaker than a checksum — say so in the tier."""
        detector = RegexDetector({"PHONE"}, phone_regions=["GB"])

        spans = detector.detect(f"call {NATIONAL['GB']}")

        assert spans[0].confidence == CONF_FUZZY

    def test_the_built_in_pattern_keeps_its_structural_tier(self) -> None:
        spans = RegexDetector({"PHONE"}).detect("call +90 532 123 45 67")

        assert spans[0].confidence == CONF_STRUCTURAL


class TestGuardBuilder:
    def test_with_phone_regions_switches_detection_over(self) -> None:
        guard = Wardcat(salt="s").add_entity(Entity.PHONE, Action.REDACT)
        text = f"call {NATIONAL['GB']} today"

        assert guard.scan(text).sanitized_text == text  # built-in pattern misses it

        guard = guard.with_phone_regions("GB")

        assert guard.scan(text).sanitized_text == "call [PHONE] today"

    def test_calling_it_empty_returns_to_the_built_in_pattern(self) -> None:
        guard = Wardcat(salt="s").add_entity(Entity.PHONE, Action.REDACT).with_phone_regions("GB")
        text = f"call {NATIONAL['GB']} today"
        assert guard.scan(text).sanitized_text == "call [PHONE] today"

        guard = guard.with_phone_regions()

        assert guard.scan(text).sanitized_text == text

    def test_codes_are_normalised(self) -> None:
        guard = Wardcat(salt="s").add_entity(Entity.PHONE, Action.REDACT).with_phone_regions(" gb ")

        assert guard.scan(f"call {NATIONAL['GB']}").sanitized_text == "call [PHONE]"

    def test_phone_stays_off_when_the_entity_is_not_enabled(self) -> None:
        guard = Wardcat(salt="s").add_entity(Entity.EMAIL, Action.REDACT).with_phone_regions("GB")

        assert guard.scan(f"call {NATIONAL['GB']}").violations == []


def test_a_missing_library_falls_back_to_the_pattern(monkeypatch, caplog) -> None:
    """Without the extra installed the guard must degrade, not crash."""
    import builtins

    real_import = builtins.__import__

    def no_phonenumbers(name, *args, **kwargs):
        if name == "phonenumbers":
            raise ImportError("simulated missing extra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_phonenumbers)
    with caplog.at_level(logging.WARNING, logger="wardcat.detectors.regex_detector"):
        detector = RegexDetector({"PHONE"}, phone_regions=["GB"])
    monkeypatch.undo()

    spans = detector.detect("call +90 532 123 45 67 and 07700 063 966")

    # The very first scan already falls back — the pattern is not skipped.
    assert [s.text for s in spans] == ["+90 532 123 45 67"]
    assert "wardcat[phone]" in caplog.text

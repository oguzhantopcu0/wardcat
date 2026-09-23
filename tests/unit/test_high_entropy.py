"""HIGH_ENTROPY_STRING: an opt-in guess at secrets with no prefix and no cue."""

from __future__ import annotations

import pytest

from wardcat import Action, Entity, Wardcat
from wardcat.detectors.regex_detector import _COMPILED, CONF_UNCUED, RegexDetector
from wardcat.utils.regex_safety import is_catastrophic

TOKEN = "q7Xk2mP9vL4nR8tY1wZ3bN6cJ5hF0dG2sA4eK9uM"  # 40 chars, mixed classes
SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SHA1 = "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12"
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"


def found(text: str) -> list[tuple[str, float]]:
    detector = RegexDetector({"HIGH_ENTROPY_STRING"})
    return [(s.text, s.confidence) for s in detector.detect(text)]


class TestWhatCounts:
    def test_a_random_token_is_found_at_the_uncued_tier(self) -> None:
        assert found(f"token: {TOKEN} ok") == [(TOKEN, CONF_UNCUED)]

    def test_a_sha256_digest_counts_but_a_sha1_does_not(self) -> None:
        assert found(f"sum {SHA256}") == [(SHA256, CONF_UNCUED)]
        assert found(f"commit {SHA1}") == []

    @pytest.mark.parametrize(
        "text",
        [
            "Supercalifragilisticexpialidocious_and_then_some",  # letters only
            "12345678901234567890123456789012345678",  # digits only
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1",  # low entropy
            "https://example.com/some/long/path/that/goes/on",  # not one token
            "org/spec/2011/FpML-5-4-0-STP-reporting/version2",  # a URL path
            "MessageReferenceNumber1234567890ABCDEF",  # words then digits
            "gaap__2023-01-31_us-gaap_instance_investment_in_equity_2023",  # a slug
            "short1Ab",
        ],
    )
    def test_ordinary_long_runs_are_not(self, text: str) -> None:
        assert found(text) == []

    def test_a_longer_run_is_taken_whole_never_sliced(self) -> None:
        run = "x" * 10 + TOKEN + "y" * 10
        assert [t for t, _ in found(run)] in ([], [run])


class TestPolicy:
    def test_off_unless_the_entity_is_enabled(self) -> None:
        guard = Wardcat(salt="s").add_entity(Entity.EMAIL, Action.REDACT)
        assert guard.scan(f"key {TOKEN}").is_clean

    def test_under_the_default_floor_it_does_not_act(self) -> None:
        guard = Wardcat(salt="s").add_entity(Entity.HIGH_ENTROPY_STRING, Action.REDACT)
        assert guard.scan(f"key {TOKEN}").is_clean

    def test_with_its_own_floor_it_acts(self) -> None:
        guard = Wardcat(salt="s").add_entity(
            Entity.HIGH_ENTROPY_STRING, Action.REDACT, min_confidence=0.7
        )
        assert guard.scan(f"key {TOKEN}").sanitized_text == "key [HIGH_ENTROPY_STRING]"

    def test_a_known_secret_shape_wins_the_overlap(self) -> None:
        guard = Wardcat(salt="s").add_entities(
            {"JWT": "redact", "HIGH_ENTROPY_STRING": {"action": "redact", "min_confidence": 0.7}}
        )
        result = guard.scan(f"bearer {JWT}")
        assert [v.entity_type for v in result.violations] == ["JWT"]


def test_the_pattern_passes_the_redos_screen() -> None:
    assert not is_catastrophic(_COMPILED["HIGH_ENTROPY_STRING"])

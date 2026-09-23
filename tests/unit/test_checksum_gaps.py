"""Troy cards, grouped TC numbers, labelled SSNs, and checksums on LLM proposals."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from wardcat.detectors.llm_detector import LLMDetector
from wardcat.detectors.regex_detector import (
    _COMPILED,
    _CUED_SSN,
    CONF_CHECKSUM,
    CONF_FUZZY,
    CONF_STRUCTURAL,
    RegexDetector,
)
from wardcat.llm.backends.base import BaseLLMBackend
from wardcat.utils.regex_safety import is_catastrophic


def found(entity: str, text: str) -> list[tuple[str, float]]:
    return [(s.text, s.confidence) for s in RegexDetector({entity}).detect(text)]


class TestTroyCards:
    @pytest.mark.parametrize(
        "text", ["Kartım 9792 8113 6631 2448, son kullanma 11/27.", "kart=9792811366312448"]
    )
    def test_a_troy_number_is_a_card(self, text: str) -> None:
        [(value, confidence)] = found("CREDIT_CARD", text)

        assert value.replace(" ", "") == "9792811366312448"
        assert confidence == CONF_CHECKSUM

    def test_it_still_has_to_pass_luhn(self) -> None:
        assert found("CREDIT_CARD", "9792 8113 6631 2449") == []


class TestGroupedTcNumbers:
    def test_the_3_3_3_2_grouping_is_found(self) -> None:
        assert found("TC_ID", "Kimlik numarası: 111 654 670 34") == [
            ("111 654 670 34", CONF_CHECKSUM)
        ]

    def test_the_whole_form_is_unchanged(self) -> None:
        assert found("TC_ID", "TC 11165467034 2024") == [("11165467034", CONF_CHECKSUM)]

    @pytest.mark.parametrize(
        "text",
        [
            "ref 111 654 670 35",  # fails the checksum
            "9 111 654 670 34",  # a slice of a longer grouped run
            "111 654 670 34 5",
            "1116 546 703 4",  # another grouping
        ],
    )
    def test_nothing_else_is(self, text: str) -> None:
        assert found("TC_ID", text) == []


class TestLabelledSsn:
    @pytest.mark.parametrize(
        ("text", "value"),
        [
            ("social security number 412 76 9038 on the intake form", "412 76 9038"),
            ("SSN: 412769038", "412769038"),
            ("Social Security No. 412 76 9038", "412 76 9038"),
        ],
    )
    def test_a_labelled_value_is_found_at_fuzzy(self, text: str, value: str) -> None:
        assert found("SSN", text) == [(value, CONF_FUZZY)]

    def test_a_dashed_ssn_is_reported_once_at_its_own_tier(self) -> None:
        assert found("SSN", "ssn is 536-90-4172") == [("536-90-4172", CONF_STRUCTURAL)]

    @pytest.mark.parametrize(
        "text", ["order 412 76 9038", "SSN 000 12 3456", "SSN 666 12 3456", "SSN 412 00 9038"]
    )
    def test_unlabelled_or_impossible_values_are_not(self, text: str) -> None:
        assert found("SSN", text) == []


def llm_spans(text: str, entity: str, value: str) -> list[str]:
    backend = MagicMock(spec=BaseLLMBackend)
    reply = json.dumps({"entities": [{"type": entity, "text": value}]})
    backend.complete.return_value = reply
    backend.complete_messages.return_value = reply
    detector = LLMDetector(backend=backend, enabled_entities={entity})
    return [s.text for s in detector.detect(text)]


class TestLlmProposalsPassChecksums:
    @pytest.mark.parametrize(
        ("entity", "value"),
        [
            ("CREDIT_CARD", "4111 1111 1111 1112"),
            ("IBAN", "TR00 0000 0000 0000 0000 0000 00"),
            ("TC_ID", "11165467035"),
        ],
    )
    def test_an_invalid_value_is_dropped(self, entity: str, value: str) -> None:
        assert llm_spans(f"value {value} here", entity, value) == []

    @pytest.mark.parametrize(
        ("entity", "value"),
        [
            ("CREDIT_CARD", "9792 8113 6631 2448"),
            ("IBAN", "TR79 0005 0168 7020 2658 0181 04"),
            ("TC_ID", "111 654 670 34"),
        ],
    )
    def test_a_valid_value_is_kept(self, entity: str, value: str) -> None:
        assert llm_spans(f"value {value} here", entity, value) == [value]


@pytest.mark.parametrize("pattern", [_CUED_SSN, _COMPILED["TC_ID"], _COMPILED["CREDIT_CARD"]])
def test_the_changed_patterns_pass_the_redos_screen(pattern) -> None:
    assert not is_catastrophic(pattern)

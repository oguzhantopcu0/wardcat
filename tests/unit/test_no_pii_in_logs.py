"""No log line, at any level, carries a value from the text being scanned."""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from wardcat import Action, Entity, Wardcat
from wardcat.detectors.llm_detector import LLMDetector
from wardcat.llm.backends.base import BaseLLMBackend

# Values chosen so the regex validators reject some (a wrong checksum is
# logged as "rejected") and accept others.
VALUES = [
    "4111 1111 1111 1112",  # fails Luhn → "format match rejected" path
    "4924 4830 2585 8831",
    "TR00 0000 0000 0000 0000 0000 00",
    "ali.veli@example.com",
    "0532 123 45 67",
    "93682697538",
]
TEXT = "Kart " + ", ".join(VALUES) + " ve tc 11165467035 son."


def assert_no_value_in(text: str) -> None:
    for value in VALUES + ["11165467035"]:
        assert value not in text, f"value {value!r} appeared in a log line"
        assert value.replace(" ", "") not in text.replace(" ", "")


class TestRegexLayer:
    def test_debug_logs_carry_no_value(self, caplog: pytest.LogCaptureFixture) -> None:
        guard = Wardcat(salt="s").add_entities(
            [Entity.CREDIT_CARD, Entity.IBAN, Entity.EMAIL, Entity.PHONE, Entity.TC_ID],
            action=Action.REDACT,
        )
        with caplog.at_level(logging.DEBUG, logger="wardcat"):
            guard.scan(TEXT)

        assert caplog.records, "expected the scan to log something at DEBUG"
        assert_no_value_in(caplog.text)


class TestLlmLayer:
    def test_hallucination_filters_and_bad_replies_log_no_value(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A reply that trips every filter: an invalid card (checksum), a badly
        # shaped IBAN (format), and then a reply that is not JSON at all.
        replies = iter(
            [
                json.dumps(
                    {
                        "entities": [
                            {"type": "CREDIT_CARD", "text": "4111 1111 1111 1112"},
                            {"type": "IBAN", "text": "ali.veli@example.com"},
                        ]
                    }
                ),
                "not json: 4924 4830 2585 8831",
            ]
        )
        backend = MagicMock(spec=BaseLLMBackend)
        backend.complete_messages.side_effect = lambda *a, **k: next(replies)
        detector = LLMDetector(backend=backend, enabled_entities={"CREDIT_CARD", "IBAN"})

        with caplog.at_level(logging.DEBUG, logger="wardcat"):
            detector.detect(TEXT)
            detector.detect(TEXT + " again")

        assert caplog.records
        assert_no_value_in(caplog.text)


@pytest.mark.ner
class TestNerLayer:
    def test_person_filters_log_no_value(self, caplog: pytest.LogCaptureFixture) -> None:
        guard = (
            Wardcat(salt="s")
            .with_ner(spacy_model="en_core_web_sm", auto_download=False)
            .add_entities([Entity.PERSON, Entity.ORG], action=Action.REDACT)
        )
        # Spans the filters reject: an address-looking "person", an all-stopword ORG.
        text = "Senior Backend Engineer met Moda Caddesi No:42 and John Smith at SSN today."
        with caplog.at_level(logging.DEBUG, logger="wardcat"):
            guard.scan(text)

        for value in ("John Smith", "Moda Caddesi", "Senior Backend Engineer"):
            assert value not in caplog.text


SUSPECT_NAMES = {
    "text",
    "value",
    "values",
    "raw",
    "reply",
    "original",
    "original_value",
    "folded_value",
    "entity_text",
    "chunk",
    "chunk_text",
    "prompt",
    "messages",
}


def _logger_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "logger"
            and node.func.attr in {"debug", "info", "warning", "error", "exception"}
        ):
            yield node


def _names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def test_no_logger_call_passes_a_text_bearing_variable() -> None:
    """A static guard for the next contributor: values must go through describe()."""
    root = Path(__file__).resolve().parents[2] / "src" / "wardcat"
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in _logger_calls(tree):
            for arg in call.args[1:]:
                # describe(value) is the sanctioned way to mention a value.
                if (
                    isinstance(arg, ast.Call)
                    and isinstance(arg.func, ast.Name)
                    and arg.func.id == "describe"
                ):
                    continue
                if isinstance(arg, ast.Name) and arg.id in SUSPECT_NAMES:
                    offenders.append(f"{path.relative_to(root)}:{call.lineno} logs {arg.id}")
                elif isinstance(arg, ast.Attribute) and arg.attr in SUSPECT_NAMES:
                    offenders.append(f"{path.relative_to(root)}:{call.lineno} logs .{arg.attr}")
                elif isinstance(arg, ast.Subscript) and _names_in(arg) & SUSPECT_NAMES:
                    offenders.append(f"{path.relative_to(root)}:{call.lineno} logs a slice")
    assert not offenders, "\n".join(offenders)

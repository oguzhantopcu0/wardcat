"""Violation.sanitized_start/end locate the replacement in sanitized_text."""

from __future__ import annotations

import random

import pytest

from wardcat import Action, Entity, Wardcat
from wardcat.core.anonymizer import Anonymizer
from wardcat.detectors.base import DetectedSpan

ACTIONS = [Action.WARN, Action.HASH, Action.REDACT, Action.MASK, Action.TOKENIZE]


def check(result) -> None:
    for v in result.violations:
        expected = v.replacement if v.replacement is not None else v.original
        assert result.sanitized_text[v.sanitized_start : v.sanitized_end] == expected
        assert result.original_text[v.start : v.end] == v.original


@pytest.mark.parametrize("action", ACTIONS)
def test_each_action_through_the_guard(action: Action) -> None:
    guard = Wardcat(salt="s").add_entities(
        [Entity.EMAIL, Entity.CREDIT_CARD, Entity.IBAN], action=action
    )
    result = guard.scan(
        "mail a@b.io then card 4111 1111 1111 1111 and iban TR33 0006 1005 1978 6457 8413 26 end"
    )
    assert len(result.violations) == 3
    check(result)


def test_reapply_recomputes_them() -> None:
    guard = Wardcat(salt="s").add_entities([Entity.EMAIL, Entity.PHONE], action=Action.WARN)
    result = guard.scan("a@b.io, Phone: 905-674-3793.")
    check(result)
    check(result.reapply(Action.REDACT))
    check(result.reapply(Action.TOKENIZE, entities=["EMAIL"]))


def test_random_span_layouts() -> None:
    rng = random.Random(7)
    for _ in range(200):
        n = rng.randint(1, 8)
        words = [f"w{i}" * rng.randint(1, 4) for i in range(n * 2)]
        text = " ".join(words)
        spans, pos = [], 0
        for i, word in enumerate(words):
            if i % 2 == 0:
                action = rng.choice(["warn", "redact", "hash", "tokenize", "mask"])
                spans.append((DetectedSpan("EMAIL", word, pos, pos + len(word), 0.97), action))
            pos += len(word) + 1
        config = {"EMAIL": {"action": "warn"}}
        # One action per span: run the anonymizer once per distinct action group
        # by giving every span the same action within a layout.
        action = rng.choice(["warn", "redact", "hash", "tokenize", "mask"])
        config["EMAIL"]["action"] = action
        sanitized, violations = Anonymizer(config, salt="s").apply(text, [s for s, _ in spans])
        for v in violations:
            expected = v.replacement if v.replacement is not None else v.original
            assert sanitized[v.sanitized_start : v.sanitized_end] == expected


def test_redacted_carries_them() -> None:
    result = Wardcat(salt="s").add_entity(Entity.EMAIL, Action.REDACT).scan("x a@b.io y")
    [v] = result.redacted()["violations"]
    assert (v["sanitized_start"], v["sanitized_end"]) == (2, 9)

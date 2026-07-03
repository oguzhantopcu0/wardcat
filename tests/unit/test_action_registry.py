"""Tests for the pluggable action registry and the separate Anonymizer stage."""

import pytest

from wardcat import Wardcat, register_action, registered_actions
from wardcat.core.actions import ActionContext, get_action
from wardcat.core.anonymizer import Anonymizer
from wardcat.detectors.base import DetectedSpan
from wardcat.exceptions import ConfigError


def test_builtin_actions_registered():
    assert {"warn", "hash", "redact", "mask"} <= registered_actions()


def test_register_and_use_custom_action():
    register_action("tokenize", lambda span, ctx: f"<TOK:{span.entity_type}>")
    assert "tokenize" in registered_actions()
    guard = Wardcat(salt="s", use_ner=False).add_entity("EMAIL", "tokenize")
    result = guard.scan("mail: a@b.com")
    assert result.sanitized_text == "mail: <TOK:EMAIL>"
    assert result.violations[0].action == "tokenize"


def test_custom_action_can_use_salt_from_context():
    register_action("salted", lambda span, ctx: f"[{ctx.salt}]")
    guard = Wardcat(salt="pepper", use_ner=False).add_entity("EMAIL", "salted")
    assert guard.scan("a@b.com").sanitized_text == "[pepper]"


def test_unknown_action_raises_with_registered_list():
    guard = Wardcat(salt="s", use_ner=False)
    with pytest.raises(ConfigError, match="Registered actions"):
        guard.add_entity("EMAIL", "does_not_exist")


def test_violation_action_compares_to_action_constant():
    from wardcat import Action

    guard = Wardcat(salt="s", use_ner=False).add_entity("CREDIT_CARD", Action.HASH)
    v = guard.scan("4111 1111 1111 1111").violations[0]
    assert v.action == Action.HASH  # str-equality still holds
    assert v.action == "hash"


# ── Anonymizer as a standalone stage (detection ⊥ anonymization) ───────────────


def test_anonymizer_runs_independently_of_detection():
    """The Anonymizer applies actions given spans — no detector/engine needed."""
    anon = Anonymizer(entity_config={"EMAIL": {"action": "redact"}}, salt="")
    spans = [DetectedSpan("EMAIL", "a@b.com", 6, 13)]
    sanitized, violations = anon.apply("mail: a@b.com", spans)
    assert sanitized == "mail: [EMAIL]"
    assert violations[0].entity_type == "EMAIL"
    assert violations[0].action == "redact"


def test_get_action_returns_callable():
    fn = get_action("redact")
    span = DetectedSpan("EMAIL", "a@b.com", 0, 7)
    assert fn(span, ActionContext(salt="")) == "[EMAIL]"

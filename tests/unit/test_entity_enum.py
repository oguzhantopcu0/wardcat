"""Tests for the Entity constants and their use in the configuration API."""

import pytest

from ai_guard import Action, Entity, LLMGuard
from ai_guard.core.models import KNOWN_ENTITY_TYPES
from ai_guard.exceptions import ConfigError


def test_entity_is_str_value():
    """Each Entity member equals its plain string form."""
    assert Entity.EMAIL == "EMAIL"
    assert Entity.CREDIT_CARD == "CREDIT_CARD"
    assert Entity.EMAIL.value == "EMAIL"


def test_known_entity_types_derived_from_enum():
    """KNOWN_ENTITY_TYPES is exactly the set of Entity values (single source)."""
    assert KNOWN_ENTITY_TYPES == frozenset(e.value for e in Entity)


def test_entity_usable_as_dict_key_like_string():
    """An Entity member keys a dict identically to its string value."""
    d = {Entity.EMAIL: 1}
    assert d["EMAIL"] == 1
    assert Entity.EMAIL in {"EMAIL"}


def test_configure_entity_accepts_entity_and_action_constants():
    guard = LLMGuard(salt="s", use_ner=False).configure_entity(
        Entity.CREDIT_CARD, action=Action.HASH
    )
    result = guard.scan("card 4532 0151 1283 0366")
    assert not result.is_clean
    v = result.violations[0]
    assert v.entity_type == "CREDIT_CARD"
    assert v.action == Action.HASH


def test_string_and_enum_are_equivalent():
    """Configuring with a string vs. an Entity/Action constant yields the same result."""
    text = "card 4532 0151 1283 0366"

    enum_guard = LLMGuard(salt="s", use_ner=False).configure_entity(
        Entity.CREDIT_CARD, action=Action.HASH
    )
    str_guard = LLMGuard(salt="s", use_ner=False).configure_entity("CREDIT_CARD", action="hash")

    assert enum_guard.scan(text).sanitized_text == str_guard.scan(text).sanitized_text


def test_configure_entities_with_entity_keys():
    guard = LLMGuard(salt="s", use_ner=False).configure_entities(
        {Entity.CREDIT_CARD: Action.HASH, Entity.EMAIL: "redact"}
    )
    result = guard.scan("card 4532 0151 1283 0366, mail a@b.com")
    types = {v.entity_type for v in result.violations}
    assert {"CREDIT_CARD", "EMAIL"} <= types


def test_invalid_action_still_raises():
    guard = LLMGuard(salt="s", use_ner=False)
    with pytest.raises(ConfigError):
        guard.configure_entity(Entity.EMAIL, action="nope")

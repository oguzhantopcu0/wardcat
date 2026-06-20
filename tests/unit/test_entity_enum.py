"""Tests for the Entity constants and their use in the configuration API."""

import warnings

import pytest

from ai_guard import Action, AIGuard, Entity
from ai_guard.core.models import KNOWN_ENTITY_TYPES
from ai_guard.exceptions import ConfigError


def test_entity_is_str_value():
    """Each Entity member equals its plain string form."""
    assert Entity.EMAIL == "EMAIL"
    assert Entity.CREDIT_CARD == "CREDIT_CARD"
    assert Entity.EMAIL.value == "EMAIL"


def test_known_entity_types_derived_from_enum_excluding_all():
    """KNOWN_ENTITY_TYPES is the set of Entity values minus the All sentinel."""
    assert KNOWN_ENTITY_TYPES == frozenset(e.value for e in Entity if e is not Entity.All)
    assert "__ALL__" not in KNOWN_ENTITY_TYPES
    assert Entity.All.value not in KNOWN_ENTITY_TYPES


def test_entity_usable_as_dict_key_like_string():
    """An Entity member keys a dict identically to its string value."""
    d = {Entity.EMAIL: 1}
    assert d["EMAIL"] == 1
    assert Entity.EMAIL in {"EMAIL"}


def test_add_entity_accepts_entity_and_action_constants():
    guard = AIGuard(salt="s", use_ner=False).add_entity(Entity.CREDIT_CARD, action=Action.HASH)
    result = guard.scan("card 4532 0151 1283 0366")
    assert not result.is_clean
    v = result.violations[0]
    assert v.entity_type == "CREDIT_CARD"
    assert v.action == Action.HASH


def test_string_and_enum_are_equivalent():
    """Configuring with a string vs. an Entity/Action constant yields the same result."""
    text = "card 4532 0151 1283 0366"

    enum_guard = AIGuard(salt="s", use_ner=False).add_entity(Entity.CREDIT_CARD, action=Action.HASH)
    str_guard = AIGuard(salt="s", use_ner=False).add_entity("CREDIT_CARD", action="hash")

    assert enum_guard.scan(text).sanitized_text == str_guard.scan(text).sanitized_text


def test_add_entities_with_entity_keys():
    guard = AIGuard(salt="s", use_ner=False).add_entities(
        {Entity.CREDIT_CARD: Action.HASH, Entity.EMAIL: "redact"}
    )
    result = guard.scan("card 4532 0151 1283 0366, mail a@b.com")
    types = {v.entity_type for v in result.violations}
    assert {"CREDIT_CARD", "EMAIL"} <= types


def test_invalid_action_still_raises():
    guard = AIGuard(salt="s", use_ner=False)
    with pytest.raises(ConfigError):
        guard.add_entity(Entity.EMAIL, action="nope")


# ---------------------------------------------------------------------------
# Entity.All — enable everything
# ---------------------------------------------------------------------------


def test_add_entity_all_enables_every_known_entity():
    guard = AIGuard(salt="s", use_ner=False).add_entity(Entity.All, action="hash")
    enabled = {name for name, cfg in guard._config["entities"].items() if cfg["enabled"]}
    # Every regex/NER-supported known entity should be enabled. (NER ones too,
    # but their engine-side enable depends on the layer; check the config flag.)
    assert KNOWN_ENTITY_TYPES <= set(guard._config["entities"])
    # A representative set is actually enabled for scanning.
    assert {"CREDIT_CARD", "EMAIL", "IBAN", "TC_ID"} <= enabled


def test_add_entity_all_then_remove_prunes_one():
    guard = (
        AIGuard(salt="s", use_ner=False)
        .add_entity(Entity.All, action="hash")
        .remove_entity(Entity.EMAIL)
    )
    result = guard.scan("card 4532 0151 1283 0366, mail a@b.com")
    types = {v.entity_type for v in result.violations}
    assert "CREDIT_CARD" in types
    assert "EMAIL" not in types


def test_add_entities_accepts_all_sentinel_in_iterable():
    guard = AIGuard(salt="s", use_ner=False).add_entities([Entity.All], action="redact")
    assert KNOWN_ENTITY_TYPES <= set(guard._config["entities"])


# ---------------------------------------------------------------------------
# remove_entity / remove_entities
# ---------------------------------------------------------------------------


def test_remove_entity_disables_detection():
    guard = AIGuard(salt="s", use_ner=False).add_entity(Entity.EMAIL, action="warn")
    assert not guard.scan("mail a@b.com").is_clean
    guard.remove_entity(Entity.EMAIL)
    assert guard.scan("mail a@b.com").is_clean


def test_remove_entity_unknown_is_noop():
    """Removing something never enabled does not raise."""
    guard = AIGuard(salt="s", use_ner=False)
    guard.remove_entity(Entity.PASSPORT)  # never added → no-op
    guard.remove_entity("NEVER_ADDED_CUSTOM")  # unknown string → no-op


def test_remove_entities_disables_many():
    guard = AIGuard(salt="s", use_ner=False).add_entities(
        ["EMAIL", "CREDIT_CARD", "IBAN"], action="hash"
    )
    guard.remove_entities([Entity.EMAIL, Entity.CREDIT_CARD])
    result = guard.scan("card 4532 0151 1283 0366, mail a@b.com")
    assert {v.entity_type for v in result.violations} == set()  # IBAN not present in text


def test_remove_entity_all_disables_everything():
    guard = AIGuard(salt="s", use_ner=False).add_entity(Entity.All, action="hash")
    guard.remove_entity(Entity.All)
    result = guard.scan("card 4532 0151 1283 0366, mail a@b.com, tc 12345678950")
    assert result.is_clean


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_add_entity_rejects_non_string_type():
    guard = AIGuard(salt="s", use_ner=False)
    with pytest.raises(ConfigError, match="must be a str or Entity"):
        guard.add_entity(123)  # type: ignore[arg-type]


def test_remove_entity_rejects_non_string_type():
    guard = AIGuard(salt="s", use_ner=False)
    with pytest.raises(ConfigError, match="must be a str or Entity"):
        guard.remove_entity(object())  # type: ignore[arg-type]


def test_add_entities_rejects_bare_string():
    guard = AIGuard(salt="s", use_ner=False)
    with pytest.raises(ConfigError, match="single str"):
        guard.add_entities("EMAIL")  # a bare string would iterate characters


def test_remove_entities_rejects_bare_entity():
    guard = AIGuard(salt="s", use_ner=False)
    with pytest.raises(ConfigError, match="single"):
        guard.remove_entities(Entity.EMAIL)


def test_set_entity_rejects_all_sentinel_directly():
    guard = AIGuard(salt="s", use_ner=False)
    with pytest.raises(ConfigError, match="Entity.All cannot be set directly"):
        guard._set_entity(Entity.All, enabled=True, action="hash", layers=None)


def test_add_entities_rejects_non_iterable():
    guard = AIGuard(salt="s", use_ner=False)
    with pytest.raises(ConfigError, match="mapping or an iterable"):
        guard.add_entities(123)  # type: ignore[arg-type]


def test_remove_entities_rejects_non_iterable():
    guard = AIGuard(salt="s", use_ner=False)
    with pytest.raises(ConfigError, match="iterable of entity types"):
        guard.remove_entities(123)  # type: ignore[arg-type]


def test_add_entities_rejects_invalid_spec_type():
    guard = AIGuard(salt="s", use_ner=False)
    with pytest.raises(ConfigError, match="Invalid spec"):
        guard.add_entities({"EMAIL": 123})  # type: ignore[dict-item]


def test_add_entities_rejects_non_string_member():
    guard = AIGuard(salt="s", use_ner=False)
    with pytest.raises(ConfigError, match="must be a str or Entity"):
        guard.add_entities([123])  # type: ignore[list-item]


def test_invalid_action_type_raises():
    guard = AIGuard(salt="s", use_ner=False)
    with pytest.raises(ConfigError, match="Invalid action"):
        guard.add_entity(Entity.EMAIL, action=999)  # type: ignore[arg-type]


def test_invalid_action_unhashable_raises():
    """An unhashable action (e.g. a list) is rejected, not propagated as TypeError."""
    guard = AIGuard(salt="s", use_ner=False)
    with pytest.raises(ConfigError, match="Invalid action"):
        guard.add_entity(Entity.EMAIL, action=["hash"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# change_entity_action
# ---------------------------------------------------------------------------


def _clean_guard():
    """A guard with no entities enabled (default config cleared)."""
    return AIGuard(salt="s", use_ner=False).remove_entity(Entity.All)


def test_change_action_updates_active_entity():
    guard = _clean_guard().add_entity(Entity.EMAIL, action="warn")
    assert guard.scan("mail a@b.com").sanitized_text == "mail a@b.com"  # warn keeps text
    guard.change_entity_action(Entity.EMAIL, Action.REDACT)
    assert guard.scan("mail a@b.com").sanitized_text == "mail [EMAIL]"


def test_change_action_returns_self_for_chaining():
    guard = _clean_guard().add_entity(Entity.EMAIL, action="warn")
    assert guard.change_entity_action(Entity.EMAIL, "redact") is guard


def test_change_action_on_never_added_raises():
    guard = _clean_guard()
    with pytest.raises(ConfigError, match="not enabled"):
        guard.change_entity_action(Entity.PASSPORT, "hash")


def test_change_action_on_removed_raises():
    guard = _clean_guard().add_entity(Entity.EMAIL, action="warn")
    guard.remove_entity(Entity.EMAIL)
    with pytest.raises(ConfigError, match="not enabled"):
        guard.change_entity_action(Entity.EMAIL, "hash")


def test_change_action_invalid_action_raises_even_when_active():
    guard = _clean_guard().add_entity(Entity.EMAIL, action="warn")
    with pytest.raises(ConfigError, match="Invalid action"):
        guard.change_entity_action(Entity.EMAIL, "explode")


def test_change_action_rejects_non_string_type():
    guard = _clean_guard().add_entity(Entity.EMAIL, action="warn")
    with pytest.raises(ConfigError, match="must be a str or Entity"):
        guard.change_entity_action(123, "hash")  # type: ignore[arg-type]


def test_change_action_all_changes_every_active_entity():
    guard = _clean_guard().add_entities(["EMAIL", "CREDIT_CARD"], action="warn")
    # warn keeps the card visible
    assert "4532" in guard.scan("card 4532 0151 1283 0366").sanitized_text
    guard.change_entity_action(Entity.All, Action.HASH)
    assert "[CREDIT_CARD:" in guard.scan("card 4532 0151 1283 0366").sanitized_text


def test_change_action_all_raises_when_nothing_active():
    guard = _clean_guard()
    with pytest.raises(ConfigError, match="no entities are currently enabled"):
        guard.change_entity_action(Entity.All, "hash")


def test_change_action_on_llm_only_entity():
    guard = _clean_guard().add_entity(Entity.SPECIAL_CATEGORY, action="redact", layers=["llm"])
    assert guard._is_entity_active("SPECIAL_CATEGORY")
    guard.change_entity_action(Entity.SPECIAL_CATEGORY, Action.WARN)
    llm_entities = guard._config["llm_detector"]["entities"]
    assert llm_entities["SPECIAL_CATEGORY"]["action"] == "warn"


# ---------------------------------------------------------------------------
# Deprecated method aliases
# ---------------------------------------------------------------------------


def test_llmguard_name_is_gone():
    """The old class name is no longer importable."""
    import ai_guard

    assert not hasattr(ai_guard, "LLMGuard")
    with pytest.raises(ImportError):
        from ai_guard import LLMGuard  # noqa: F401


def test_configure_entity_alias_warns_but_works():
    guard = AIGuard(salt="s", use_ner=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        guard.configure_entity(Entity.EMAIL, action="warn")
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert not guard.scan("mail a@b.com").is_clean

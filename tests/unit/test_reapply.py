"""Tests for ScanResult.reapply — deriving a different action from one scan."""

from __future__ import annotations

import pytest

from wardcat import Action, ConfigError, Entity, Wardcat

TEXT = "mail bob@acme.com card 4111 1111 1111 1111"


def _guard(action=Action.REDACT):
    return Wardcat(salt="my-secret").add_entities([Entity.EMAIL, Entity.CREDIT_CARD], action=action)


def test_reapply_matches_a_natively_configured_scan():
    """reapply(action) must equal scanning natively with that action."""
    result = _guard().scan(TEXT)
    for action in (Action.WARN, Action.HASH, Action.REDACT, Action.MASK):
        native = _guard(action).scan(TEXT).sanitized_text
        assert result.reapply(action).sanitized_text == native, action


def test_reapply_accepts_action_name_string():
    result = _guard().scan(TEXT)
    assert result.reapply("mask").sanitized_text == result.reapply(Action.MASK).sanitized_text


def test_reapply_reuses_the_guard_salt_for_hash():
    hashed = _guard().scan(TEXT).reapply(Action.HASH).sanitized_text
    same = _guard().scan(TEXT).reapply(Action.HASH).sanitized_text
    different_salt = (
        Wardcat(salt="OTHER")
        .add_entities([Entity.EMAIL, Entity.CREDIT_CARD], action=Action.REDACT)
        .scan(TEXT)
        .reapply(Action.HASH)
        .sanitized_text
    )
    assert hashed == same  # deterministic under the same salt
    assert hashed != different_salt  # salt actually participates


def test_reapply_entities_subset_leaves_other_pii_untouched():
    result = _guard().scan(TEXT)
    out = result.reapply(Action.MASK, entities=[Entity.EMAIL])
    assert "b**@acme.com" in out.sanitized_text  # email masked
    assert "4111 1111 1111 1111" in out.sanitized_text  # card untouched
    assert {v.entity_type for v in out.violations} == {"EMAIL"}


def test_reapply_undetected_entity_is_not_an_error():
    # IBAN was never enabled/detected; asking for it just yields nothing to do.
    out = _guard().scan(TEXT).reapply(Action.HASH, entities=["IBAN"])
    assert out.violations == []
    assert out.sanitized_text == TEXT  # nothing anonymized


def test_reapply_rejects_unknown_action():
    result = _guard().scan(TEXT)
    with pytest.raises(ConfigError, match="unknown action"):
        result.reapply("bogus")


def test_reapply_returns_a_fresh_result_without_mutating_the_original():
    result = _guard().scan(TEXT)
    before = result.sanitized_text
    derived = result.reapply(Action.MASK)
    assert result.sanitized_text == before  # original untouched
    assert derived is not result
    assert derived.reapply(Action.REDACT).sanitized_text == "mail [EMAIL] card [CREDIT_CARD]"


def test_reapply_on_clean_text_stays_clean():
    out = _guard().scan("nothing sensitive").reapply(Action.HASH)
    assert out.is_clean
    assert out.sanitized_text == "nothing sensitive"

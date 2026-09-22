"""One entity can have its own confidence floor."""

from __future__ import annotations

import pytest

from wardcat import Action, ConfigError, Entity, Wardcat

# A bare ABA routing number: passes the mod-10 check but has no keyword, so the
# regex layer scores it 0.70 — under the default floor of 0.8.
BARE_ROUTING = "transfer via 021000021 today"


def test_the_global_floor_drops_an_uncued_checksum_match() -> None:
    guard = Wardcat(salt="s").add_entity(Entity.BANK_ROUTING, Action.REDACT)
    assert guard.scan(BARE_ROUTING).is_clean


def test_an_entity_floor_lets_it_through_without_lowering_the_rest() -> None:
    guard = (
        Wardcat(salt="s")
        .add_entity(Entity.BANK_ROUTING, Action.REDACT, min_confidence=0.6)
        .add_entity(Entity.NHS_NUMBER, Action.REDACT)
    )
    assert guard.scan(BARE_ROUTING).sanitized_text == "transfer via [BANK_ROUTING] today"
    # A bare NHS-shaped number still falls under the global 0.8.
    assert guard.scan("ref 943 476 5919").is_clean


def test_the_floor_survives_a_re_add_that_does_not_mention_it() -> None:
    guard = (
        Wardcat(salt="s")
        .add_entity(Entity.BANK_ROUTING, Action.REDACT, min_confidence=0.6)
        .add_entity(Entity.BANK_ROUTING, Action.HASH)
    )
    assert guard.entity_policy(detailed=True)["BANK_ROUTING"] == {
        "action": "hash",
        "min_confidence": 0.6,
    }
    assert not guard.scan(BARE_ROUTING).is_clean


def test_add_entities_spec_form() -> None:
    guard = Wardcat(salt="s").add_entities(
        {"BANK_ROUTING": {"action": "redact", "min_confidence": 0.6}, "EMAIL": "redact"}
    )
    assert not guard.scan(BARE_ROUTING).is_clean
    assert guard.entity_policy() == {"BANK_ROUTING": "redact", "EMAIL": "redact"}


def test_yaml_form(tmp_path) -> None:
    cfg = tmp_path / "p.yaml"
    cfg.write_text(
        "entities:\n  BANK_ROUTING: {enabled: true, action: redact, min_confidence: 0.6}\n"
    )
    assert not Wardcat(config_path=str(cfg), salt="s").scan(BARE_ROUTING).is_clean


@pytest.mark.parametrize("bad", [1.5, -0.1, "high"])
def test_an_invalid_floor_is_refused(bad, tmp_path) -> None:
    with pytest.raises(ConfigError):
        Wardcat(salt="s").add_entity(Entity.EMAIL, Action.REDACT, min_confidence=bad)
    cfg = tmp_path / "p.yaml"
    cfg.write_text(
        f"entities:\n  EMAIL: {{enabled: true, action: redact, min_confidence: {bad!r}}}\n"
    )
    with pytest.raises(ConfigError):
        Wardcat(config_path=str(cfg), salt="s")

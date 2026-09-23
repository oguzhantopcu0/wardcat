"""Presets: an entity → action mapping, nothing switched on, nothing promised."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from wardcat import Action, ConfigError, Entity, Wardcat, supported_presets
from wardcat.core.models import KNOWN_ENTITY_TYPES
from wardcat.core.registry import NER_ENTITIES, REGEX_ENTITIES
from wardcat.presets import PRESETS, get_preset


@pytest.mark.parametrize("name", supported_presets())
class TestEveryPreset:
    def test_entities_and_actions_are_known(self, name: str) -> None:
        from wardcat.core.actions import registered_actions

        preset = get_preset(name)
        assert set(preset.entities) <= KNOWN_ENTITY_TYPES
        assert set(preset.entities.values()) <= registered_actions()
        assert preset.covers and preset.not_covered

    def test_needs_layers_matches_the_registry(self, name: str) -> None:
        preset = get_preset(name)
        expected = {
            "ner" if e in NER_ENTITIES else "llm"
            for e in preset.entities
            if e not in REGEX_ENTITIES
        }
        assert preset.needs_layers == expected

    def test_with_preset_sets_exactly_that_policy(self, name: str) -> None:
        guard = Wardcat(salt="s").with_preset(name)
        preset = get_preset(name)
        # entity_policy() lists what is active; an LLM-only entity such as
        # SPECIAL_CATEGORY is configured but inactive until with_llm() runs.
        llm_only = {e for e in preset.entities if e not in REGEX_ENTITIES | NER_ENTITIES}
        expected = {e: a for e, a in preset.entities.items() if e not in llm_only}
        assert guard.entity_policy() == expected
        for entity in llm_only:
            assert guard._config["llm_detector"]["entities"][entity]["enabled"] is True

    def test_no_layer_is_switched_on(self, name: str) -> None:
        guard = Wardcat(salt="s").with_preset(name)
        assert guard._config.get("use_ner", False) is False
        assert not guard._config["llm_detector"].get("enabled", False)

    def test_the_guide_lists_the_same_entities(self, name: str) -> None:
        guide = Path(__file__).resolve().parents[2] / "docs" / "guide" / "presets.md"
        # utf-8 named: the arrows in the guide are not in Windows' default codepage.
        text = guide.read_text(encoding="utf-8")
        section = re.search(rf"^## `{name}`\n(.*?)(?=^## |\Z)", text, re.S | re.M)
        assert section, f"presets.md has no section for {name}"
        listed = set(re.findall(r"`([A-Z][A-Z0-9_]+)` → `(\w+)`", section.group(1)))
        assert listed == set(get_preset(name).entities.items())


class TestComposition:
    def test_builders_after_the_preset_adjust_it(self) -> None:
        guard = (
            Wardcat(salt="s")
            .with_preset("kvkk")
            .change_entity_action(Entity.PHONE, Action.REDACT)
            .remove_entity(Entity.VEHICLE_PLATE)
        )
        policy = guard.entity_policy()
        assert policy["PHONE"] == "redact"
        assert "VEHICLE_PLATE" not in policy

    def test_order_does_not_matter_for_a_layer(self) -> None:
        a = Wardcat(salt="s").with_preset("secrets_only").add_entity(Entity.EMAIL, Action.REDACT)
        b = Wardcat(salt="s").add_entity(Entity.EMAIL, Action.REDACT).with_preset("secrets_only")
        assert a.entity_policy() == b.entity_policy()

    def test_yaml_preset_with_overrides(self, tmp_path) -> None:
        cfg = tmp_path / "p.yaml"
        cfg.write_text(
            "preset: pci_dss\nentities:\n  CREDIT_CARD: {enabled: true, action: redact}\n"
        )
        guard = Wardcat(config_path=str(cfg), salt="s")
        policy = guard.entity_policy()
        assert policy["CREDIT_CARD"] == "redact"  # the file wins
        assert policy["IBAN"] == "hash"  # the preset fills the rest
        assert "preset" not in guard._config

    def test_unknown_preset(self, tmp_path) -> None:
        with pytest.raises(ConfigError, match="Unknown preset"):
            Wardcat(salt="s").with_preset("sox")
        cfg = tmp_path / "p.yaml"
        cfg.write_text("preset: sox\n")
        with pytest.raises(ConfigError, match="Unknown preset"):
            Wardcat(config_path=str(cfg))

    def test_a_preset_scans(self) -> None:
        guard = Wardcat(salt="s").with_preset("pci_dss")
        result = guard.scan(
            "card 4111 1111 1111 1111 and token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1In0.abc"
        )
        assert "[CREDIT_CARD:" in result.sanitized_text
        assert "[JWT]" in result.sanitized_text

    def test_presets_are_read_only(self) -> None:
        with pytest.raises(TypeError):
            PRESETS["kvkk"].entities["EMAIL"] = "warn"  # type: ignore[index]

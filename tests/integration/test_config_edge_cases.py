"""
Config loader and LLMGuard configuration deep tests.

Scope:
  - Malformed / missing YAML
  - Unknown entity types
  - Invalid action value
  - Deep-merge correctness
  - Programmatic API edge cases
  - Salt reset
  - Entity disabled → no detection
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ai_guard import LLMGuard
from ai_guard.config.loader import DEFAULT_CONFIG, _deep_merge, load_config


# ══════════════════════════════════════════════════════════════════════════
# load_config — deep-merge logic
# ══════════════════════════════════════════════════════════════════════════

class TestDeepMerge:
    def test_override_scalar(self):
        result = _deep_merge({"a": 1}, {"a": 2})
        assert result["a"] == 2

    def test_adds_new_key(self):
        result = _deep_merge({"a": 1}, {"b": 2})
        assert result["a"] == 1
        assert result["b"] == 2

    def test_nested_dict_merged_not_replaced(self):
        base     = {"entities": {"EMAIL": {"enabled": True, "action": "warn"}}}
        override = {"entities": {"EMAIL": {"action": "hash"}}}
        result   = _deep_merge(base, override)
        # action was overridden but enabled came from the default
        assert result["entities"]["EMAIL"]["action"]   == "hash"
        assert result["entities"]["EMAIL"]["enabled"]  is True

    def test_new_nested_key_added(self):
        base     = {"entities": {"EMAIL": {"enabled": True, "action": "warn"}}}
        override = {"entities": {"TC_ID": {"enabled": True, "action": "hash"}}}
        result   = _deep_merge(base, override)
        assert "EMAIL" in result["entities"]
        assert "TC_ID" in result["entities"]

    def test_base_not_mutated(self):
        base = {"a": {"x": 1}}
        _deep_merge(base, {"a": {"y": 2}})
        assert "y" not in base["a"]


class TestLoadConfig:
    def test_no_path_returns_defaults(self):
        cfg = load_config()
        assert cfg == DEFAULT_CONFIG

    def test_unknown_top_level_key_allowed(self, tmp_path: Path):
        f = tmp_path / "cfg.yaml"
        f.write_text(yaml.dump({"unknown_key": "value", "salt": "test"}))
        cfg = load_config(f)
        assert cfg["unknown_key"] == "value"
        assert cfg["salt"] == "test"

    def test_extra_entity_type_added(self, tmp_path: Path):
        f = tmp_path / "cfg.yaml"
        f.write_text(yaml.dump({"entities": {"CUSTOM_ENTITY": {"enabled": True, "action": "warn"}}}))
        cfg = load_config(f)
        assert "CUSTOM_ENTITY" in cfg["entities"]
        assert "EMAIL" in cfg["entities"]   # default preserved

    def test_partial_entity_override(self, tmp_path: Path):
        """When only 'action' is overridden, 'enabled' should remain at default."""
        f = tmp_path / "cfg.yaml"
        f.write_text(yaml.dump({"entities": {"EMAIL": {"action": "hash"}}}))
        cfg = load_config(f)
        assert cfg["entities"]["EMAIL"]["action"]  == "hash"
        assert cfg["entities"]["EMAIL"]["enabled"] is True  # default

    def test_empty_yaml_returns_defaults(self, tmp_path: Path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        assert load_config(f) == DEFAULT_CONFIG

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/tmp/kesinlikle_yok_xyz.yaml")

    def test_salt_override_from_yaml(self, tmp_path: Path):
        f = tmp_path / "cfg.yaml"
        f.write_text(yaml.dump({"salt": "yaml-tuz"}))
        cfg = load_config(f)
        assert cfg["salt"] == "yaml-tuz"


# ══════════════════════════════════════════════════════════════════════════
# LLMGuard programmatic API edge cases
# ══════════════════════════════════════════════════════════════════════════

class TestProgrammaticAPIEdgeCases:
    def test_configure_unknown_entity_type(self):
        """Unknown entity type should be configurable (no effect without regex/NER)."""
        guard = LLMGuard(use_ner=False)
        guard.configure_entity("MY_CUSTOM_PII", enabled=True, action="warn")
        # should not raise, scan should work
        result = guard.scan("bazı metin")
        assert result is not None

    def test_disable_all_entities(self):
        guard = LLMGuard(use_ner=False)
        for entity in ["CREDIT_CARD", "EMAIL", "PHONE", "IBAN", "IP_ADDRESS",
                        "TC_ID", "ADDRESS", "POSTAL_CODE"]:
            guard.configure_entity(entity, enabled=False)
        text = "4111111111111111 a@b.com 12345678950 TR330006100519786457841326"
        result = guard.scan(text)
        assert result.is_clean

    def test_configure_entity_returns_self(self):
        guard = LLMGuard(use_ner=False)
        returned = guard.configure_entity("EMAIL", enabled=True, action="warn")
        assert returned is guard

    def test_set_salt_returns_self(self):
        guard = LLMGuard(use_ner=False)
        assert guard.set_salt("yeni-tuz") is guard

    def test_salt_in_constructor_overrides_yaml(self, tmp_path: Path):
        f = tmp_path / "cfg.yaml"
        f.write_text(yaml.dump({"salt": "yaml-tuz"}))
        guard = LLMGuard(config_path=f, salt="constructor-tuz", use_ner=False)
        assert guard._config["salt"] == "constructor-tuz"

    def test_set_salt_empty_string(self):
        guard = LLMGuard(use_ner=False, salt="mevcut")
        guard.set_salt("")
        assert guard._config["salt"] == ""

    def test_reconfigure_entity_midstream(self):
        guard = LLMGuard(use_ner=False)
        guard.configure_entity("EMAIL", enabled=True, action="warn")
        r1 = guard.scan("a@b.com")
        guard.configure_entity("EMAIL", enabled=True, action="hash")
        r2 = guard.scan("a@b.com")

        assert "a@b.com" in r1.sanitized_text      # warn → unchanged
        assert "a@b.com" not in r2.sanitized_text  # hash → changed


# ══════════════════════════════════════════════════════════════════════════
# Entity enabled/disabled behavior
# ══════════════════════════════════════════════════════════════════════════

class TestEntityEnablement:
    @pytest.mark.parametrize("entity,text", [
        ("CREDIT_CARD", "4111111111111111"),
        ("EMAIL",       "a@b.com"),
        ("PHONE",       "0532 111 22 33"),
        ("IBAN",        "TR330006100519786457841326"),
        ("IP_ADDRESS",  "192.168.1.1"),
        ("TC_ID",       "12345678950"),
    ])
    def test_disabled_entity_not_detected(self, entity, text):
        guard = LLMGuard(use_ner=False)
        guard.configure_entity(entity, enabled=False)
        result = guard.scan(text)
        assert not any(v.entity_type == entity for v in result.violations), \
            f"{entity} was detected despite being disabled"

    @pytest.mark.parametrize("entity,text", [
        ("CREDIT_CARD", "4111111111111111"),
        ("EMAIL",       "a@b.com"),
        ("TC_ID",       "12345678950"),
    ])
    def test_re_enabling_entity_works(self, entity, text):
        guard = LLMGuard(use_ner=False)
        guard.configure_entity(entity, enabled=False)
        guard.configure_entity(entity, enabled=True, action="warn")
        result = guard.scan(text)
        assert any(v.entity_type == entity for v in result.violations), \
            f"{entity} was not re-enabled"


# ══════════════════════════════════════════════════════════════════════════
# YAML + Programmatic API ordering
# ══════════════════════════════════════════════════════════════════════════

class TestYAMLAndProgrammaticCombined:
    def test_yaml_then_programmatic_override(self, tmp_path: Path):
        f = tmp_path / "cfg.yaml"
        f.write_text(yaml.dump({
            "entities": {"EMAIL": {"enabled": True, "action": "warn"}}
        }))
        guard = LLMGuard(config_path=f, use_ner=False)
        guard.configure_entity("EMAIL", action="hash")   # override

        result = guard.scan("a@b.com")
        v = next(v for v in result.violations if v.entity_type == "EMAIL")
        from ai_guard.core.models import Action
        assert v.action == Action.HASH

    def test_yaml_salt_and_programmatic_entity(self, tmp_path: Path):
        f = tmp_path / "cfg.yaml"
        f.write_text(yaml.dump({"salt": "yaml-tuz"}))
        guard = LLMGuard(config_path=f, use_ner=False)
        guard.configure_entity("TC_ID", enabled=True, action="hash")

        r1 = guard.scan("TC: 12345678950")
        guard2 = LLMGuard(config_path=f, salt="farkli-tuz", use_ner=False)
        guard2.configure_entity("TC_ID", enabled=True, action="hash")
        r2 = guard2.scan("TC: 12345678950")

        # Different salt → different hash
        assert r1.sanitized_text != r2.sanitized_text

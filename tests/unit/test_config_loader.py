import tempfile
from pathlib import Path

import pytest
import yaml

from ai_guard.config.loader import DEFAULT_CONFIG, load_config


def test_load_defaults_when_no_path():
    config = load_config()
    assert "entities" in config
    assert config["entities"]["CREDIT_CARD"]["action"] == "hash"


def test_yaml_overrides_default(tmp_path: Path):
    override = {
        "salt": "test-salt",
        "entities": {
            "EMAIL": {"enabled": False, "action": "hash"},
        },
    }
    cfg_file = tmp_path / "policy.yaml"
    cfg_file.write_text(yaml.dump(override))

    config = load_config(cfg_file)

    assert config["salt"] == "test-salt"
    assert config["entities"]["EMAIL"]["enabled"] is False
    assert config["entities"]["EMAIL"]["action"] == "hash"
    # Other entities should come from defaults
    assert config["entities"]["CREDIT_CARD"]["action"] == "hash"


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/tmp/bu_dosya_yok_12345.yaml")


def test_empty_yaml_keeps_defaults(tmp_path: Path):
    cfg_file = tmp_path / "empty.yaml"
    cfg_file.write_text("")
    config = load_config(cfg_file)
    assert config["entities"] == DEFAULT_CONFIG["entities"]


def test_load_bundled_default():
    """load_config('default') should load the bundled default.yaml."""
    config = load_config("default")
    assert "entities" in config
    assert "EMAIL" in config["entities"]


def test_missing_yaml_file_raises():
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config("/tmp/does_not_exist_xyz.yaml")


def test_deep_copy_handles_list():
    from ai_guard.config.loader import _deep_copy
    original = {"key": [1, 2, {"nested": True}]}
    copied = _deep_copy(original)
    copied["key"][0] = 99
    assert original["key"][0] == 1  # original unchanged


# ── G2a: max_text_bytes configurable ──────────────────────────────────────

def test_default_max_text_bytes():
    config = load_config()
    assert config["max_text_bytes"] == 500_000


def test_custom_max_text_bytes_from_yaml(tmp_path: Path):
    override = {"max_text_bytes": 100_000}
    cfg_file = tmp_path / "policy.yaml"
    cfg_file.write_text(yaml.dump(override))
    config = load_config(cfg_file)
    assert config["max_text_bytes"] == 100_000


def test_max_text_bytes_enforced_by_engine(tmp_path: Path):
    """Engine should use the configured max_text_bytes, not the hardcoded default."""
    from ai_guard import LLMGuard
    cfg = {"max_text_bytes": 10}
    cfg_file = tmp_path / "policy.yaml"
    cfg_file.write_text(yaml.dump(cfg))
    guard = LLMGuard(config_path=str(cfg_file), use_ner=False)
    with pytest.raises(ValueError, match="too large"):
        guard.scan("x" * 11)


# ── G6: custom_patterns validation ────────────────────────────────────────

def test_custom_patterns_default_is_empty():
    config = load_config()
    assert config["custom_patterns"] == {}


def test_custom_patterns_loaded_from_yaml(tmp_path: Path):
    override = {
        "custom_patterns": {
            "EMPLOYEE_ID": {
                "pattern": r"\bEMP-\d{6}\b",
                "action": "hash",
            }
        }
    }
    cfg_file = tmp_path / "policy.yaml"
    cfg_file.write_text(yaml.dump(override))
    config = load_config(cfg_file)
    assert "EMPLOYEE_ID" in config["custom_patterns"]
    assert config["custom_patterns"]["EMPLOYEE_ID"]["action"] == "hash"


def test_custom_patterns_invalid_regex_raises(tmp_path: Path):
    override = {
        "custom_patterns": {
            "BAD": {"pattern": "[invalid", "action": "warn"},
        }
    }
    cfg_file = tmp_path / "policy.yaml"
    cfg_file.write_text(yaml.dump(override))
    with pytest.raises(ValueError, match="invalid regex"):
        load_config(cfg_file)


def test_custom_patterns_missing_pattern_key_raises(tmp_path: Path):
    override = {
        "custom_patterns": {
            "NO_PATTERN": {"action": "warn"},
        }
    }
    cfg_file = tmp_path / "policy.yaml"
    cfg_file.write_text(yaml.dump(override))
    with pytest.raises(ValueError, match="missing required 'pattern' key"):
        load_config(cfg_file)


def test_custom_patterns_invalid_action_raises(tmp_path: Path):
    override = {
        "custom_patterns": {
            "MY_PATTERN": {"pattern": r"\btest\b", "action": "delete"},
        }
    }
    cfg_file = tmp_path / "policy.yaml"
    cfg_file.write_text(yaml.dump(override))
    with pytest.raises(ValueError, match="Invalid action"):
        load_config(cfg_file)


def test_custom_patterns_detected_in_scan(tmp_path: Path):
    """End-to-end: custom_patterns in YAML config flows through to detection."""
    from ai_guard import LLMGuard
    override = {
        "custom_patterns": {
            "EMPLOYEE_ID": {
                "pattern": r"\bEMP-\d{6}\b",
                "action": "warn",
            }
        }
    }
    cfg_file = tmp_path / "policy.yaml"
    cfg_file.write_text(yaml.dump(override))
    guard = LLMGuard(config_path=str(cfg_file), use_ner=False)
    result = guard.scan("employee EMP-123456 logged in")
    assert any(v.entity_type == "EMPLOYEE_ID" for v in result.violations)

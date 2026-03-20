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

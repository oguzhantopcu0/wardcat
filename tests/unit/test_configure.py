"""Tests for the layer-aware filter API: configure_entity(layers=...) and configure_entities()."""

import pytest

from ai_guard import LLMGuard, turkish_entities


@pytest.fixture
def guard():
    # NER disabled → tests run even without SpaCy installed
    return LLMGuard(use_ner=False)


# ---------------------------------------------------------------------------
# configure_entity(layers=...)
# ---------------------------------------------------------------------------


def test_configure_entity_default_layers_auto_detect(guard):
    """With no layers, an entity goes to every layer that supports it."""
    guard.configure_entity("EMAIL", enabled=True, action="redact")
    assert guard._config["entities"]["EMAIL"] == {"enabled": True, "action": "redact"}


def test_configure_entity_llm_only_layer(guard):
    """layers=['llm'] writes to the LLM set, NOT the regex/NER enabled set."""
    guard.configure_entity("SPECIAL_CATEGORY", enabled=True, action="redact", layers=["llm"])
    # action recorded so the engine can apply it, but regex/NER enabled is False
    assert guard._config["entities"]["SPECIAL_CATEGORY"] == {
        "enabled": False,
        "action": "redact",
    }
    assert guard._config["llm_detector"]["entities"]["SPECIAL_CATEGORY"] == {
        "enabled": True,
        "action": "redact",
    }


def test_configure_entity_regex_only_layer(guard):
    """layers=['regex'] enables regex but does not touch the LLM set."""
    llm_before = dict(guard._config.get("llm_detector", {}).get("entities", {}))
    guard.configure_entity("EMAIL", enabled=True, action="hash", layers=["regex"])
    assert guard._config["entities"]["EMAIL"] == {"enabled": True, "action": "hash"}
    # LLM set is left exactly as it was — regex-only targeting must not edit it
    assert guard._config.get("llm_detector", {}).get("entities", {}) == llm_before


def test_configure_entity_invalid_layer_raises(guard):
    with pytest.raises(ValueError, match="Invalid layer"):
        guard.configure_entity("EMAIL", layers=["bogus"])


def test_configure_entity_invalid_action_raises(guard):
    with pytest.raises(ValueError, match="Invalid action"):
        guard.configure_entity("EMAIL", action="explode")


def test_configure_entity_returns_self_for_chaining(guard):
    result = guard.configure_entity("EMAIL", action="redact")
    assert result is guard


def test_configure_entity_disable(guard):
    guard.configure_entity("EMAIL", enabled=False)
    assert guard._config["entities"]["EMAIL"]["enabled"] is False


# ---------------------------------------------------------------------------
# configure_entities() — batch
# ---------------------------------------------------------------------------


def test_configure_entities_iterable(guard):
    guard.configure_entities(["EMAIL", "CREDIT_CARD", "IBAN"], action="redact")
    for ent in ("EMAIL", "CREDIT_CARD", "IBAN"):
        assert guard._config["entities"][ent] == {"enabled": True, "action": "redact"}


def test_configure_entities_group_helper(guard):
    guard.configure_entities(turkish_entities(), action="hash")
    assert guard._config["entities"]["TC_ID"] == {"enabled": True, "action": "hash"}
    assert guard._config["entities"]["EMAIL"] == {"enabled": True, "action": "hash"}


def test_configure_entities_mapping_name_to_action(guard):
    guard.configure_entities({"EMAIL": "warn", "CREDIT_CARD": "hash"})
    assert guard._config["entities"]["EMAIL"]["action"] == "warn"
    assert guard._config["entities"]["CREDIT_CARD"]["action"] == "hash"


def test_configure_entities_mapping_per_entity_spec(guard):
    guard.configure_entities(
        {
            "CREDIT_CARD": "hash",
            "EMAIL": {"action": "mask"},
            "SPECIAL_CATEGORY": {"action": "redact", "layers": ["llm"]},
        }
    )
    assert guard._config["entities"]["CREDIT_CARD"] == {"enabled": True, "action": "hash"}
    assert guard._config["entities"]["EMAIL"] == {"enabled": True, "action": "mask"}
    # llm-only entity: not enabled for regex/NER, but present in LLM set
    assert guard._config["entities"]["SPECIAL_CATEGORY"]["enabled"] is False
    assert guard._config["llm_detector"]["entities"]["SPECIAL_CATEGORY"]["enabled"] is True


def test_configure_entities_top_level_defaults_apply(guard):
    """Top-level action/layers act as defaults for entries without their own."""
    guard.configure_entities({"EMAIL": {}, "CREDIT_CARD": "hash"}, action="redact")
    assert guard._config["entities"]["EMAIL"]["action"] == "redact"
    assert guard._config["entities"]["CREDIT_CARD"]["action"] == "hash"


def test_configure_entities_enabled_false(guard):
    guard.configure_entities(["EMAIL", "PHONE"], enabled=False)
    assert guard._config["entities"]["EMAIL"]["enabled"] is False
    assert guard._config["entities"]["PHONE"]["enabled"] is False


def test_configure_entities_invalid_spec_type_raises(guard):
    with pytest.raises(ValueError, match="Invalid spec"):
        guard.configure_entities({"EMAIL": 123})


def test_configure_entities_returns_self_for_chaining(guard):
    result = guard.configure_entities(["EMAIL"])
    assert result is guard


# ---------------------------------------------------------------------------
# End-to-end: configured filters actually run
# ---------------------------------------------------------------------------


def test_configure_entities_end_to_end_detection():
    guard = LLMGuard(use_ner=False)
    guard.configure_entities(["EMAIL", "CREDIT_CARD"], action="redact")
    result = guard.scan("Mail: john@x.com kart 4111 1111 1111 1111")
    assert "[EMAIL]" in result.sanitized_text
    assert "[CREDIT_CARD]" in result.sanitized_text


def test_configure_entity_llm_only_does_not_enable_regex_scan():
    """An llm-only entity must not be detected by the regex layer."""
    guard = LLMGuard(use_ner=False)
    guard.configure_entity("EMAIL", enabled=True, action="redact", layers=["llm"])
    # EMAIL only assigned to llm; regex layer should not flag it
    result = guard.scan("Mail: john@x.com")
    assert "[EMAIL]" not in result.sanitized_text

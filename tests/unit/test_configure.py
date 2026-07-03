"""Tests for the layer-aware filter API: add_entity(layers=...) and add_entities()."""

import pytest

from wardcat import Wardcat, turkish_entities


@pytest.fixture
def guard():
    # NER disabled → tests run even without SpaCy installed
    return Wardcat(use_ner=False)


# ---------------------------------------------------------------------------
# add_entity(layers=...)
# ---------------------------------------------------------------------------


def test_add_entity_default_layers_auto_detect(guard):
    """With no layers, an entity goes to every layer that supports it."""
    guard.add_entity("EMAIL", action="redact")
    assert guard._config["entities"]["EMAIL"] == {"enabled": True, "action": "redact"}


def test_add_entity_llm_only_layer(guard):
    """layers=['llm'] writes to the LLM set, NOT the regex/NER enabled set."""
    guard.add_entity("SPECIAL_CATEGORY", action="redact", layers=["llm"])
    # action recorded so the engine can apply it, but regex/NER enabled is False
    assert guard._config["entities"]["SPECIAL_CATEGORY"] == {
        "enabled": False,
        "action": "redact",
    }
    assert guard._config["llm_detector"]["entities"]["SPECIAL_CATEGORY"] == {
        "enabled": True,
        "action": "redact",
    }


def test_add_entity_regex_only_layer(guard):
    """layers=['regex'] enables regex but does not touch the LLM set."""
    llm_before = dict(guard._config.get("llm_detector", {}).get("entities", {}))
    guard.add_entity("EMAIL", action="hash", layers=["regex"])
    assert guard._config["entities"]["EMAIL"] == {"enabled": True, "action": "hash"}
    # LLM set is left exactly as it was — regex-only targeting must not edit it
    assert guard._config.get("llm_detector", {}).get("entities", {}) == llm_before


def test_add_entity_invalid_layer_raises(guard):
    with pytest.raises(ValueError, match="Invalid layer"):
        guard.add_entity("EMAIL", layers=["bogus"])


def test_add_entity_invalid_action_raises(guard):
    with pytest.raises(ValueError, match="Invalid action"):
        guard.add_entity("EMAIL", action="explode")


def test_add_entity_returns_self_for_chaining(guard):
    result = guard.add_entity("EMAIL", action="redact")
    assert result is guard


def test_remove_entity_disables(guard):
    guard.add_entity("EMAIL")
    guard.remove_entity("EMAIL")
    assert guard._config["entities"]["EMAIL"]["enabled"] is False


# ---------------------------------------------------------------------------
# add_entities() — batch
# ---------------------------------------------------------------------------


def test_add_entities_iterable(guard):
    guard.add_entities(["EMAIL", "CREDIT_CARD", "IBAN"], action="redact")
    for ent in ("EMAIL", "CREDIT_CARD", "IBAN"):
        assert guard._config["entities"][ent] == {"enabled": True, "action": "redact"}


def test_add_entities_group_helper(guard):
    guard.add_entities(turkish_entities(), action="hash")
    assert guard._config["entities"]["TC_ID"] == {"enabled": True, "action": "hash"}
    assert guard._config["entities"]["EMAIL"] == {"enabled": True, "action": "hash"}


def test_add_entities_mapping_name_to_action(guard):
    guard.add_entities({"EMAIL": "warn", "CREDIT_CARD": "hash"})
    assert guard._config["entities"]["EMAIL"]["action"] == "warn"
    assert guard._config["entities"]["CREDIT_CARD"]["action"] == "hash"


def test_add_entities_mapping_per_entity_spec(guard):
    guard.add_entities(
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


def test_add_entities_top_level_defaults_apply(guard):
    """Top-level action/layers act as defaults for entries without their own."""
    guard.add_entities({"EMAIL": {}, "CREDIT_CARD": "hash"}, action="redact")
    assert guard._config["entities"]["EMAIL"]["action"] == "redact"
    assert guard._config["entities"]["CREDIT_CARD"]["action"] == "hash"


def test_remove_entities_disables(guard):
    guard.add_entities(["EMAIL", "PHONE"])
    guard.remove_entities(["EMAIL", "PHONE"])
    assert guard._config["entities"]["EMAIL"]["enabled"] is False
    assert guard._config["entities"]["PHONE"]["enabled"] is False


def test_add_entities_invalid_spec_type_raises(guard):
    with pytest.raises(ValueError, match="Invalid spec"):
        guard.add_entities({"EMAIL": 123})


def test_add_entities_returns_self_for_chaining(guard):
    result = guard.add_entities(["EMAIL"])
    assert result is guard


# ---------------------------------------------------------------------------
# End-to-end: configured filters actually run
# ---------------------------------------------------------------------------


def test_add_entities_end_to_end_detection():
    guard = Wardcat(use_ner=False)
    guard.add_entities(["EMAIL", "CREDIT_CARD"], action="redact")
    result = guard.scan("Mail: john@x.com kart 4111 1111 1111 1111")
    assert "[EMAIL]" in result.sanitized_text
    assert "[CREDIT_CARD]" in result.sanitized_text


def test_add_entity_llm_only_does_not_enable_regex_scan():
    """An llm-only entity must not be detected by the regex layer."""
    guard = Wardcat(use_ner=False)
    guard.add_entity("EMAIL", action="redact", layers=["llm"])
    # EMAIL only assigned to llm; regex layer should not flag it
    result = guard.scan("Mail: john@x.com")
    assert "[EMAIL]" not in result.sanitized_text

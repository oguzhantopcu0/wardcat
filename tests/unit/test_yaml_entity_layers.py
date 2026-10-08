"""A YAML policy honours ``layers:`` and says when it holds a secret.

Until 1.3.0 an entity entry's ``layers`` key was accepted and ignored: a policy
limiting EMAIL to the LLM layer still had the regex layer mask it, with nothing
to say so. Any key the loader does not know in an entity entry is now an error,
and ``layers`` is applied exactly as ``add_entity(layers=...)`` applies it.
"""

from __future__ import annotations

import logging

import pytest

from wardcat import ConfigError, Wardcat


def policy(tmp_path, text: str) -> str:
    path = tmp_path / "policy.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _llm_entities(guard: Wardcat) -> dict:
    return guard._config.get("llm_detector", {}).get("entities", {})


class TestLayers:
    def test_llm_only_entity_is_not_masked_by_regex(self, tmp_path) -> None:
        g = Wardcat(
            config_path=policy(
                tmp_path, ("entities:\n  EMAIL: {enabled: true, action: redact, layers: [llm]}\n")
            )
        )
        assert g.scan("a@b.io").sanitized_text == "a@b.io"
        assert _llm_entities(g)["EMAIL"] == {"enabled": True, "action": "redact"}

    def test_yaml_matches_the_api(self, tmp_path) -> None:
        from_yaml = Wardcat(
            config_path=policy(
                tmp_path,
                (
                    "entities:\n"
                    "  EMAIL: {enabled: true, action: redact, layers: [llm]}\n"
                    "  CREDIT_CARD: {enabled: true, action: hash, layers: [regex, llm]}\n"
                ),
            )
        )
        from_api = (
            Wardcat()
            .add_entity("EMAIL", "redact", layers=["llm"])
            .add_entity("CREDIT_CARD", "hash", layers=["regex", "llm"])
        )
        for name in ("EMAIL", "CREDIT_CARD"):
            assert (
                from_yaml._config["entities"][name]["enabled"]
                == (from_api._config["entities"][name]["enabled"])
            )
            assert _llm_entities(from_yaml)[name] == _llm_entities(from_api)[name]

    def test_regex_layer_still_masks(self, tmp_path) -> None:
        g = Wardcat(
            config_path=policy(
                tmp_path, ("entities:\n  EMAIL: {enabled: true, action: redact, layers: [regex]}\n")
            )
        )
        assert g.scan("a@b.io").sanitized_text == "[EMAIL]"

    @pytest.mark.parametrize("layers", ["llm", "[]", "[magic]", "[1]"])
    def test_a_bad_layers_value_is_an_error(self, tmp_path, layers) -> None:
        with pytest.raises(ConfigError, match="layer"):
            Wardcat(
                config_path=policy(
                    tmp_path,
                    (f"entities:\n  EMAIL: {{enabled: true, action: redact, layers: {layers}}}\n"),
                )
            )


class TestUnknownEntityKeys:
    def test_a_typo_is_an_error_not_ignored(self, tmp_path) -> None:
        with pytest.raises(ConfigError, match="Unknown key"):
            Wardcat(
                config_path=policy(
                    tmp_path,
                    ("entities:\n  EMAIL: {enabled: true, action: redact, layer: [llm]}\n"),
                )
            )

    def test_llm_entities_take_only_enabled_and_action(self, tmp_path) -> None:
        with pytest.raises(ConfigError, match="Unknown key"):
            Wardcat(
                config_path=policy(
                    tmp_path,
                    (
                        "llm_detector:\n  entities:\n    PERSON: {enabled: true, action: hash, layers: [llm]}\n"
                    ),
                )
            )


class TestSecretsInTheFile:
    def test_a_literal_salt_warns_without_its_value(self, tmp_path, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="wardcat"):
            Wardcat(
                config_path=policy(
                    tmp_path,
                    ('salt: "do-not-ship-me"\nentities:\n  EMAIL: {enabled: true, action: hash}\n'),
                )
            )
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "salt in plain text" in messages
        assert "do-not-ship-me" not in messages

    def test_a_literal_api_key_warns(self, tmp_path, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="wardcat"):
            Wardcat(
                config_path=policy(
                    tmp_path,
                    (
                        'llm_detector:\n  api_key: "sk-secret-value"\n'
                        "entities:\n  EMAIL: {enabled: true, action: redact}\n"
                    ),
                )
            )
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "llm_detector.api_key" in messages and "sk-secret-value" not in messages

    def test_no_warning_without_secrets(self, tmp_path, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="wardcat"):
            Wardcat(config_path=policy(tmp_path, "entities:\n  EMAIL: {enabled: true}\n"))
        assert "plain text" not in " ".join(r.getMessage() for r in caplog.records)

    def test_check_config_shows_the_warning(self, tmp_path, capsys) -> None:
        from wardcat.cli import main

        path = policy(tmp_path, 'salt: "x"\nentities:\n  EMAIL: {enabled: true, action: hash}\n')
        assert main(["check-config", path]) == 0
        assert "salt in plain text" in capsys.readouterr().err

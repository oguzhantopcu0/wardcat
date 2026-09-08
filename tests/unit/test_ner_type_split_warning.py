"""The NER layer moved two kinds of span into types of their own.

Place names left ``ADDRESS`` for ``LOCATION``; nationality, religious and
political groups left ``ORG`` for ``NRP``. Both are corrections — a city is not
a street address, and a nationality is not a company — but a configuration
written before them keeps working and quietly covers less, which is the failure
mode that does not announce itself. It is said once, at scan time.
"""

from __future__ import annotations

import logging

import pytest

from wardcat import Action, Entity, Wardcat

pytestmark = pytest.mark.ner

MODEL = "en_core_web_sm"
TEXT = "The office is in Germany. She is Kurdish."


def _guard(*entities):
    guard = Wardcat(salt="s").with_ner(spacy_model=MODEL, auto_download=False)
    for entity in entities:
        guard.add_entity(entity, Action.REDACT)
    return guard


class TestTheWarning:
    def test_address_without_location_is_told(self, caplog):
        with caplog.at_level(logging.WARNING, logger="wardcat.guard"):
            _guard(Entity.ADDRESS).scan(TEXT)
        assert "ADDRESS no longer covers" in caplog.text
        assert "LOCATION" in caplog.text

    def test_org_without_nrp_is_told(self, caplog):
        with caplog.at_level(logging.WARNING, logger="wardcat.guard"):
            _guard(Entity.ORG).scan(TEXT)
        assert "ORG no longer covers" in caplog.text
        assert "NRP" in caplog.text

    def test_said_once_per_guard(self, caplog):
        guard = _guard(Entity.ADDRESS)
        with caplog.at_level(logging.WARNING, logger="wardcat.guard"):
            guard.scan(TEXT)
            guard.scan(TEXT)
        assert caplog.text.count("ADDRESS no longer covers") == 1

    def test_quiet_once_the_new_type_is_enabled(self, caplog):
        with caplog.at_level(logging.WARNING, logger="wardcat.guard"):
            _guard(Entity.ADDRESS, Entity.LOCATION).scan(TEXT)
        assert "no longer covers" not in caplog.text

    def test_quiet_when_the_ner_layer_is_off(self, caplog):
        """Nothing was being covered by the NER layer, so nothing was lost."""
        guard = Wardcat(salt="s").add_entity(Entity.ADDRESS, Action.REDACT)
        with caplog.at_level(logging.WARNING, logger="wardcat.guard"):
            guard.scan(TEXT)
        assert "no longer covers" not in caplog.text

    def test_quiet_for_a_caller_who_brought_their_own_policy(self, tmp_path, caplog):
        """A YAML policy is a deliberate choice, like the other warnings here."""
        cfg = tmp_path / "p.yaml"
        cfg.write_text(
            f'use_ner: true\nspacy_model: "{MODEL}"\n'
            "entities:\n  ADDRESS:\n    enabled: true\n    action: redact\n",
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING, logger="wardcat.guard"):
            Wardcat(config_path=str(cfg)).scan(TEXT)
        assert "no longer covers" not in caplog.text


class TestTheCoverageItWarnsAbout:
    def test_a_place_name_needs_location_now(self):
        assert _guard(Entity.ADDRESS).scan(TEXT).sanitized_text == TEXT
        assert "[LOCATION]" in _guard(Entity.ADDRESS, Entity.LOCATION).scan(TEXT).sanitized_text

    def test_a_nationality_needs_nrp_now(self):
        assert "[NRP]" in _guard(Entity.NRP).scan(TEXT).sanitized_text

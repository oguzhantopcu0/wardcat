"""A layer that could not be built is reported on every result, not only in a log.

The configuration guide promises that a non-empty ``warnings`` means a scan was degraded. That
held for a layer failing mid-scan, but a layer that failed while the guard was
being built — a SpaCy model that would not load, SpaCy not installed at all, the
``phonenumbers`` extra missing — was only logged, and every scan came back clean
with nothing in ``warnings``.

None of these tests need SpaCy: model loading is replaced, so they behave the
same in a minimal install as in the development environment.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from unittest.mock import MagicMock

import pytest

from wardcat import Entity, Wardcat
from wardcat.core.engine import DetectionEngine


@pytest.fixture
def ner_loads(monkeypatch):
    """Make every SpaCy model load succeed without SpaCy."""
    monkeypatch.setattr("wardcat.detectors.ner_detector._load_model", lambda name: MagicMock())
    monkeypatch.setattr("wardcat.guard._resolve_spacy_model", lambda name: name)


@pytest.fixture
def ner_fails(monkeypatch):
    def fail(name):
        raise OSError(f"[E050] Can't find model '{name}'.")

    monkeypatch.setattr("wardcat.detectors.ner_detector._load_model", fail)
    monkeypatch.setattr("wardcat.guard._resolve_spacy_model", lambda name: name)


def _ner_guard(model: str = "xx_missing") -> Wardcat:
    return (
        Wardcat()
        .with_ner(spacy_model=model, auto_download=False)
        .add_entity(Entity.PERSON, action="redact")
        .add_entity(Entity.EMAIL, action="redact")
    )


class TestNERModelThatDoesNotLoad:
    def test_every_result_says_the_layer_did_not_run(self, ner_fails):
        guard = _ner_guard()
        for _ in range(2):
            result = guard.scan("John Smith, john@example.com")
            assert result.sanitized_text == "John Smith, [EMAIL]"  # regex still ran
            assert any(
                "NERDetector did not run" in w and "xx_missing" in w for w in result.warnings
            )
            assert any("PERSON" in w and "detects nothing" in w for w in result.warnings)

    def test_spacy_not_installed_is_reported(self, monkeypatch):
        def no_spacy(name):
            raise ModuleNotFoundError("No module named 'spacy'")

        monkeypatch.setattr("wardcat.detectors.ner_detector._load_model", no_spacy)
        monkeypatch.setattr("wardcat.guard._resolve_spacy_model", lambda name: name)
        result = _ner_guard("en_core_web_sm").scan("John Smith called.")
        assert result.is_clean
        assert any("No module named 'spacy'" in w for w in result.warnings)

    def test_one_failing_model_does_not_claim_the_layer_is_empty(self, monkeypatch):
        def load(name):
            if name == "de_core_news_sm":
                raise OSError("missing")
            return MagicMock()

        monkeypatch.setattr("wardcat.detectors.ner_detector._load_model", load)
        monkeypatch.setattr("wardcat.guard._resolve_spacy_model", lambda name: name)
        guard = (
            Wardcat()
            .with_ner(spacy_model=["en_core_web_sm", "de_core_news_sm"], auto_download=False)
            .add_entity(Entity.PERSON, action="redact")
        )
        warnings = guard.scan("hello").warnings
        assert any("de_core_news_sm" in w for w in warnings)
        assert not any("detects nothing" in w for w in warnings)

    def test_async_and_batch_results_carry_it_too(self, ner_fails):
        guard = _ner_guard()
        assert any("did not run" in w for w in asyncio.run(guard.scan_async("x")).warnings)
        for result in guard.scan_batch(["a", "b"]):
            assert any("did not run" in w for w in result.warnings)
        for result in asyncio.run(guard.scan_batch_async(["a", "b"])):
            assert any("did not run" in w for w in result.warnings)

    def test_a_batch_item_that_fails_keeps_the_warning(self, ner_fails):
        guard = _ner_guard()
        guard._engine._anonymizer.apply = MagicMock(side_effect=RuntimeError("boom"))
        (result,) = guard.scan_batch(["John"])
        assert result.scan_error == "RuntimeError: boom"
        assert any("did not run" in w for w in result.warnings)

    def test_the_warning_is_not_repeated_within_one_result(self, ner_fails):
        warnings = _ner_guard().scan("x").warnings
        assert len(warnings) == len(set(warnings))


class TestNERModelSubstitution:
    def test_a_different_language_is_called_out(self, monkeypatch):
        monkeypatch.setattr("wardcat.detectors.ner_detector._load_model", lambda n: MagicMock())
        monkeypatch.setattr("wardcat.guard._resolve_spacy_model", lambda n: "tr_core_news_md")
        (warning,) = _ner_guard("de_core_news_sm").scan("Hans Müller").warnings
        assert "'tr_core_news_md' instead" in warning
        assert "different language" in warning
        assert "python -m spacy download de_core_news_sm" in warning

    def test_same_language_substitution_is_reported_without_the_language_note(self, monkeypatch):
        monkeypatch.setattr("wardcat.detectors.ner_detector._load_model", lambda n: MagicMock())
        monkeypatch.setattr("wardcat.guard._resolve_spacy_model", lambda n: "en_core_web_sm")
        (warning,) = _ner_guard("en_core_web_lg").scan("John Smith").warnings
        assert "'en_core_web_sm' instead" in warning
        assert "different language" not in warning


def test_a_healthy_setup_has_no_warnings(ner_loads):
    assert _ner_guard("en_core_web_sm").scan("John Smith, john@example.com").warnings == []


def test_missing_phonenumbers_is_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "phonenumbers", None)
    guard = Wardcat().add_entity(Entity.PHONE, action="redact").with_phone_regions("GB")
    result = guard.scan("call +90 532 123 45 67")
    assert result.sanitized_text == "call [PHONE]"  # the built-in pattern took over
    assert any("phonenumbers" in w for w in result.warnings)


def test_an_invalid_denylist_pattern_reaching_the_engine_is_reported():
    """Validation stops this at the guard; the engine still must not hide it."""
    engine = DetectionEngine({"denylist": [{"pattern": "[bad", "entity_type": "X"}]}, [])
    assert any("not valid regex" in w for w in engine.scan("x").warnings)


def test_scan_batch_logs_the_one_time_configuration_warnings(caplog):
    """scan_batch called the engine directly and skipped what scan() logs."""
    guard = Wardcat(salt="s").add_entity(Entity.PERSON)  # no layer detects PERSON
    with caplog.at_level(logging.WARNING, logger="wardcat.guard"):
        guard.scan_batch(["John Smith", "Jane Doe"])
    assert len([r for r in caplog.records if "no active layer" in r.message]) == 1

"""Tests for ner/downloader.py and Wardcat language selection / auto-download."""

from __future__ import annotations

import pytest

from wardcat import Wardcat
from wardcat.ner import downloader


class TestIsInstalled:
    def test_returns_bool(self):
        assert isinstance(downloader.is_installed("en_core_web_sm"), bool)

    def test_unknown_model_not_installed(self):
        assert downloader.is_installed("zz_not_a_real_model_xyz") is False


class TestEnsureModel:
    def test_missing_without_auto_download_returns_false(self):
        assert downloader.ensure_model("zz_not_a_real_model_xyz", auto_download=False) is False

    def test_missing_auto_download_invokes_download(self, monkeypatch):
        calls = {}

        def fake_download(model_name, *, verbose=False):
            calls["model"] = model_name

        monkeypatch.setattr(downloader, "download_model", fake_download)
        monkeypatch.setattr(
            downloader, "is_installed", lambda n: n in calls.get("installed", set())
        )

        # First is_installed → False, download called, second is_installed → still False
        result = downloader.ensure_model("de_core_news_sm", auto_download=True)
        assert calls["model"] == "de_core_news_sm"
        assert result is False  # our fake never marks it installed


class TestDownloadModelGuards:
    def test_incompatible_model_raises(self):
        pytest.importorskip("spacy")
        # tr_core_news_trf is flagged incompatible in the catalog — must raise
        # before any network/subprocess work.
        with pytest.raises(RuntimeError, match="not compatible"):
            downloader.download_model("tr_core_news_trf")


class TestGuardLanguageSelection:
    def test_language_resolves_model_name(self):
        g = Wardcat(language="de", spacy_size="md", use_ner=False)
        assert g._config["spacy_model"] == "de_core_news_md"

    def test_language_default_size_sm(self):
        g = Wardcat(language="fr", use_ner=False)
        assert g._config["spacy_model"] == "fr_core_news_sm"

    def test_language_implies_auto_download(self):
        g = Wardcat(language="en", use_ner=False)
        assert g._config.get("spacy_auto_download") is True

    def test_explicit_auto_download_off(self):
        g = Wardcat(language="en", use_ner=False, spacy_auto_download=False)
        assert g._config.get("spacy_auto_download") is None

    def test_uppercase_language_code(self):
        g = Wardcat(language="DE", use_ner=False)
        assert g._config["spacy_model"].startswith("de_")

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            Wardcat(language="zz", use_ner=False)

    def test_no_language_keeps_default_model(self):
        g = Wardcat(use_ner=False)
        assert g._config.get("spacy_model", "en_core_web_sm") == "en_core_web_sm"
        assert g._config.get("spacy_auto_download") is None

    def test_auto_download_attempted_on_rebuild(self, monkeypatch):
        """When NER is on and auto-download set, rebuild calls ensure_model."""
        seen = {}

        def fake_ensure(model_name, *, auto_download=False, verbose=False):
            seen["model"] = model_name
            seen["auto"] = auto_download
            return False  # pretend it could not be installed → NER degrades gracefully

        monkeypatch.setattr(downloader, "ensure_model", fake_ensure)
        # use_ner True triggers the NER branch; ensure_model is called there
        Wardcat(language="de", spacy_size="md", use_ner=True).add_entity("PERSON")
        assert seen["model"] == "de_core_news_md"
        assert seen["auto"] is True


class TestGuardMultiLanguage:
    def test_list_resolves_one_model_per_language(self):
        g = Wardcat(language=["de", "fr"], use_ner=False)
        assert g._config["spacy_models"] == ["de_core_news_sm", "fr_core_news_sm"]
        # primary model stays consistent with the single-model field
        assert g._config["spacy_model"] == "de_core_news_sm"

    def test_single_str_still_populates_models_list(self):
        g = Wardcat(language="de", use_ner=False)
        assert g._config["spacy_models"] == ["de_core_news_sm"]

    def test_duplicate_languages_deduped(self):
        g = Wardcat(language=["de", "de"], use_ner=False)
        assert g._config["spacy_models"] == ["de_core_news_sm"]

    def test_size_applies_to_all_languages(self):
        g = Wardcat(language=["de", "fr"], spacy_size="md", use_ner=False)
        assert g._config["spacy_models"] == ["de_core_news_md", "fr_core_news_md"]

    def test_unsupported_language_in_list_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            Wardcat(language=["de", "zz"], use_ner=False)

    def test_list_implies_auto_download(self):
        g = Wardcat(language=["de", "fr"], use_ner=False)
        assert g._config.get("spacy_auto_download") is True

    def test_loads_one_detector_per_installed_model(self):
        """End-to-end: each language with an installed model yields its own detector."""
        pytest.importorskip("spacy")
        import spacy.util

        from wardcat.detectors.ner_detector import NERDetector

        installed = set(spacy.util.get_installed_models())
        if not {"en_core_web_sm", "tr_core_news_md"} <= installed:
            pytest.skip("en_core_web_sm and tr_core_news_md must be installed")

        # Detection is opt-in: enable a NER entity first, then each language's
        # model yields its own detector.
        g = Wardcat(language=["en", "tr"], spacy_auto_download=False)
        g.add_entity("PERSON", action="redact")
        ner_detectors = [d for d in g._detectors if isinstance(d, NERDetector)]
        assert len(ner_detectors) == 2

        result = g.scan("John Smith ve Ahmet Yılmaz toplantıdaydı.")
        persons = {v.original for v in result.violations if v.entity_type == "PERSON"}
        assert "John Smith" in persons  # English model
        assert "Ahmet Yılmaz" in persons  # Turkish model

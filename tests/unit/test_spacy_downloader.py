"""Tests for ner/downloader.py and LLMGuard language selection / auto-download."""

from __future__ import annotations

import pytest

from ai_guard import LLMGuard
from ai_guard.ner import downloader


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
        g = LLMGuard(language="de", spacy_size="md", use_ner=False)
        assert g._config["spacy_model"] == "de_core_news_md"

    def test_language_default_size_sm(self):
        g = LLMGuard(language="fr", use_ner=False)
        assert g._config["spacy_model"] == "fr_core_news_sm"

    def test_language_implies_auto_download(self):
        g = LLMGuard(language="en", use_ner=False)
        assert g._config.get("spacy_auto_download") is True

    def test_explicit_auto_download_off(self):
        g = LLMGuard(language="en", use_ner=False, spacy_auto_download=False)
        assert g._config.get("spacy_auto_download") is None

    def test_uppercase_language_code(self):
        g = LLMGuard(language="DE", use_ner=False)
        assert g._config["spacy_model"].startswith("de_")

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            LLMGuard(language="zz", use_ner=False)

    def test_no_language_keeps_default_model(self):
        g = LLMGuard(use_ner=False)
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
        LLMGuard(language="de", spacy_size="md", use_ner=True)
        assert seen["model"] == "de_core_news_md"
        assert seen["auto"] is True

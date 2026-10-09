"""Tests for ner/downloader.py and Wardcat language selection / auto-download."""

from __future__ import annotations

import pytest

from wardcat import Wardcat
from wardcat.exceptions import ModelDownloadError
from wardcat.ner import downloader


class TestModelNameValidation:
    """download_model must reject names it would otherwise pass to a subprocess."""

    @pytest.mark.parametrize("name", ["de_core_news_md", "en_core_web_sm", "tr_core_news_lg"])
    def test_valid_model_names_accepted(self, name):
        downloader._validate_model_name(name)  # does not raise

    @pytest.mark.parametrize(
        "name",
        ["../evil", "x; rm -rf /", "http://evil.example/pkg", "foo-bar", "", "en core web"],
    )
    def test_malicious_or_malformed_names_rejected(self, name):
        with pytest.raises(ModelDownloadError):
            downloader._validate_model_name(name)

    def test_download_model_rejects_bad_name_before_any_subprocess(self, monkeypatch):
        import subprocess

        def _boom(*a, **k):  # pragma: no cover - must never be reached
            raise AssertionError("subprocess must not run for an invalid model name")

        monkeypatch.setattr(subprocess, "run", _boom)
        with pytest.raises(ModelDownloadError):
            downloader.download_model("../evil")


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
    def test_incompatible_model_raises(self, monkeypatch):
        """The guard must refuse before any network or subprocess work.

        The catalog entry is fabricated rather than naming a real model: this
        test used to pin tr_core_news_trf, which outlived its flag — the model
        loads fine on current SpaCy, so the test was passing on a stale claim
        instead of on the guard.
        """
        pytest.importorskip("spacy")
        from wardcat.ner.spacy_catalog import SpacyModelInfo

        blocked = SpacyModelInfo(
            name="xx_core_news_sm",
            language="Example",
            lang_code="xx",
            size="sm",
            ram_mb=1,
            description="fixture",
            note="pinned to an older SpaCy",
            spacy_compat=">=3.4,<3.5",
            incompatible=True,
        )
        monkeypatch.setattr(downloader, "get_spacy_model", lambda name: blocked)
        with pytest.raises(RuntimeError, match="not compatible"):
            downloader.download_model("xx_core_news_sm")


class TestGuardLanguageSelection:
    def test_language_resolves_model_name(self):
        g = Wardcat().with_ner(language="de", spacy_size="md")
        assert g._config["spacy_model"] == "de_core_news_md"

    def test_language_default_size_sm(self):
        g = Wardcat().with_ner(language="fr")
        assert g._config["spacy_model"] == "fr_core_news_sm"

    def test_language_implies_auto_download(self):
        g = Wardcat().with_ner(language="en")
        assert g._config.get("spacy_auto_download") is True

    def test_explicit_auto_download_off(self):
        g = Wardcat().with_ner(language="en", auto_download=False)
        assert g._config.get("spacy_auto_download") is None

    def test_uppercase_language_code(self):
        g = Wardcat().with_ner(language="DE")
        assert g._config["spacy_model"].startswith("de_")

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            Wardcat().with_ner(language="zz")

    def test_no_language_keeps_default_model(self):
        g = Wardcat()
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
        Wardcat().with_ner(language="de", spacy_size="md").add_entity("PERSON")
        assert seen["model"] == "de_core_news_md"
        assert seen["auto"] is True


class TestGuardMultiLanguage:
    def test_list_resolves_one_model_per_language(self):
        g = Wardcat().with_ner(language=["de", "fr"])
        assert g._config["spacy_models"] == ["de_core_news_sm", "fr_core_news_sm"]
        # primary model stays consistent with the single-model field
        assert g._config["spacy_model"] == "de_core_news_sm"

    def test_single_str_still_populates_models_list(self):
        g = Wardcat().with_ner(language="de")
        assert g._config["spacy_models"] == ["de_core_news_sm"]

    def test_duplicate_languages_deduped(self):
        g = Wardcat().with_ner(language=["de", "de"])
        assert g._config["spacy_models"] == ["de_core_news_sm"]

    def test_size_applies_to_all_languages(self):
        g = Wardcat().with_ner(language=["de", "fr"], spacy_size="md")
        assert g._config["spacy_models"] == ["de_core_news_md", "fr_core_news_md"]

    def test_unsupported_language_in_list_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            Wardcat().with_ner(language=["de", "zz"])

    def test_list_implies_auto_download(self):
        g = Wardcat().with_ner(language=["de", "fr"])
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
        g = Wardcat().with_ner(language=["en", "tr"], auto_download=False)
        g.add_entity("PERSON", action="redact")
        ner_detectors = [d for d in g._detectors if isinstance(d, NERDetector)]
        assert len(ner_detectors) == 2

        result = g.scan("John Smith ve Ahmet Yılmaz toplantıdaydı.")
        persons = {v.original for v in result.violations if v.entity_type == "PERSON"}
        assert "John Smith" in persons  # English model
        assert "Ahmet Yılmaz" in persons  # Turkish model


class TestInstallsIntoThisPython:
    """uv without --python installs into whatever venv the cwd or VIRTUAL_ENV names,
    so a model "downloaded" from one environment was missing in the one asking."""

    def _capture(self, monkeypatch, *, pip_ok: bool) -> list[list[str]]:
        import subprocess
        import sys as _sys

        from wardcat.ner import downloader

        calls: list[list[str]] = []

        class Done:
            def __init__(self, code: int) -> None:
                self.returncode = code

        def run(cmd, **kwargs):
            calls.append(list(cmd))
            if cmd[:3] == [_sys.executable, "-m", "pip"]:
                return Done(0 if pip_ok else 1)
            return Done(0)

        monkeypatch.setattr(subprocess, "run", run)
        monkeypatch.setattr(downloader.shutil, "which", lambda name: "/usr/bin/uv")
        return calls

    def test_uv_targets_the_running_interpreter_for_a_wheel_model(self, monkeypatch) -> None:
        import sys as _sys

        pytest.importorskip("spacy")
        from wardcat.ner.downloader import download_model

        calls = self._capture(monkeypatch, pip_ok=False)
        download_model("tr_core_news_md")
        uv = [c for c in calls if c[0] == "/usr/bin/uv"]
        assert uv and all(c[3:5] == ["--python", _sys.executable] for c in uv)

    def test_uv_targets_the_running_interpreter_for_a_github_model(self, monkeypatch) -> None:
        import sys as _sys

        pytest.importorskip("spacy")
        from wardcat.ner.downloader import download_model

        calls = self._capture(monkeypatch, pip_ok=False)
        download_model("en_core_web_sm")
        uv = [c for c in calls if c[0] == "/usr/bin/uv"]
        assert uv and all(c[3:5] == ["--python", _sys.executable] for c in uv)

    def test_import_caches_are_refreshed(self, monkeypatch) -> None:
        import importlib

        pytest.importorskip("spacy")
        from wardcat.ner.downloader import download_model

        self._capture(monkeypatch, pip_ok=True)
        refreshed = []
        monkeypatch.setattr(importlib, "invalidate_caches", lambda: refreshed.append(True))
        download_model("tr_core_news_md")
        assert refreshed


class TestIsInstalledSeesNewPackages:
    def test_an_importable_model_counts_even_if_spacy_has_not_listed_it(self, monkeypatch) -> None:
        import importlib.util

        from wardcat.ner.downloader import is_installed

        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a: object() if name == "xx_fresh_model_sm" else real(name, *a),
        )
        assert is_installed("xx_fresh_model_sm") is True

    @pytest.mark.parametrize("bad", ["os.path", "../x", "", "a b"])
    def test_an_invalid_name_is_never_looked_up(self, bad) -> None:
        from wardcat.ner.downloader import is_installed

        assert is_installed(bad) is False

"""Tests for the public exception hierarchy and the errors that raise it."""

from __future__ import annotations

import pytest

from ai_guard import (
    AIGuardError,
    ConfigError,
    LLMGuard,
    ModelDownloadError,
    UnsupportedLanguageError,
)
from ai_guard.config.loader import validate_config


class TestHierarchy:
    def test_config_error_is_aiguard_and_value_error(self):
        assert issubclass(ConfigError, AIGuardError)
        assert issubclass(ConfigError, ValueError)  # backward compatibility

    def test_model_download_error_is_aiguard_and_runtime_error(self):
        assert issubclass(ModelDownloadError, AIGuardError)
        assert issubclass(ModelDownloadError, RuntimeError)  # backward compatibility

    def test_unsupported_language_is_config_error(self):
        assert issubclass(UnsupportedLanguageError, ConfigError)


class TestConfigErrorRaised:
    def test_invalid_entity_action_raises_config_error(self):
        with pytest.raises(ConfigError, match="Invalid action"):
            validate_config({"entities": {"EMAIL": {"action": "explode"}}})

    def test_invalid_llm_entity_action_raises_config_error(self):
        # Previously this slipped through load-time validation and only failed at scan.
        with pytest.raises(ConfigError, match="llm_detector.entities"):
            validate_config({"llm_detector": {"entities": {"EMAIL": {"action": "boom"}}}})

    def test_invalid_backend_raises_config_error(self):
        with pytest.raises(ConfigError, match="Invalid LLM backend"):
            validate_config({"llm_detector": {"backend": "magic"}})

    def test_still_catchable_as_value_error(self):
        # Existing `except ValueError` code must keep working.
        with pytest.raises(ValueError):
            LLMGuard(use_ner=False).configure_entity("EMAIL", action="nope")

    def test_catchable_as_aiguard_error(self):
        with pytest.raises(AIGuardError):
            LLMGuard(use_ner=False).configure_entity("EMAIL", layers=["bogus"])


class TestUnsupportedLanguage:
    def test_unsupported_language_raises_specific_type(self):
        with pytest.raises(UnsupportedLanguageError):
            LLMGuard(language="zz", use_ner=False)

    def test_still_catchable_as_value_error(self):
        with pytest.raises(ValueError):
            LLMGuard(language="zz", use_ner=False)


class TestModelDownloadError:
    def test_incompatible_model_raises_model_download_error(self):
        pytest.importorskip("spacy")
        from ai_guard.ner.downloader import download_model

        with pytest.raises(ModelDownloadError, match="not compatible"):
            download_model("tr_core_news_trf")

    def test_still_catchable_as_runtime_error(self):
        pytest.importorskip("spacy")
        from ai_guard.ner.downloader import download_model

        with pytest.raises(RuntimeError):
            download_model("tr_core_news_trf")

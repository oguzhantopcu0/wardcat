"""Tests for explicit NER model selection (no default model; Language enum; errors)."""

import pytest

from ai_guard import AIGuard, Language
from ai_guard.exceptions import ConfigError, UnsupportedLanguageError

# ---------------------------------------------------------------------------
# NER is off by default; no default model is shipped
# ---------------------------------------------------------------------------


def test_bare_guard_has_ner_off():
    guard = AIGuard(salt="s")
    assert guard._config["use_ner"] is False
    assert not guard._config.get("spacy_models")


def test_use_ner_true_without_model_raises():
    with pytest.raises(ConfigError, match="requires a SpaCy model"):
        AIGuard(salt="s", use_ner=True)


def test_use_ner_false_with_language_stays_off_but_resolves_model():
    # Explicit use_ner=False is respected even when a language is given;
    # the model is still resolved into config (no download, NER not built).
    guard = AIGuard(salt="s", language="de", spacy_size="md", use_ner=False)
    assert guard._config["use_ner"] is False
    assert guard._config["spacy_models"] == ["de_core_news_md"]


# ---------------------------------------------------------------------------
# Language enum (documented selection) — equivalent to the bare code
# ---------------------------------------------------------------------------


def test_language_enum_is_iso_code():
    assert Language.EN == "en"
    assert Language.DE.value == "de"


def test_language_enum_resolves_like_string():
    g_enum = AIGuard(salt="s", language=Language.DE, use_ner=False)
    g_str = AIGuard(salt="s", language="de", use_ner=False)
    assert g_enum._config["spacy_models"] == g_str._config["spacy_models"]


def test_language_list_with_enums_multilingual():
    guard = AIGuard(salt="s", language=[Language.DE, Language.FR], use_ner=False)
    assert guard._config["spacy_models"] == ["de_core_news_sm", "fr_core_news_sm"]


def test_unsupported_language_raises():
    with pytest.raises(UnsupportedLanguageError):
        AIGuard(salt="s", language="zz", use_ner=False)


# ---------------------------------------------------------------------------
# Explicit spacy_model: single, multiple, implies NER on
# ---------------------------------------------------------------------------


def test_explicit_model_implies_ner_on():
    guard = AIGuard(salt="s", spacy_model="en_core_web_sm", spacy_auto_download=False)
    assert guard._config["use_ner"] is True
    assert guard._config["spacy_models"] == ["en_core_web_sm"]


def test_multiple_explicit_models():
    guard = AIGuard(
        salt="s",
        spacy_model=["en_core_web_sm", "de_core_news_sm"],
        use_ner=False,
    )
    assert guard._config["spacy_models"] == ["en_core_web_sm", "de_core_news_sm"]


def test_explicit_models_deduped():
    guard = AIGuard(
        salt="s",
        spacy_model=["en_core_web_sm", "en_core_web_sm"],
        use_ner=False,
    )
    assert guard._config["spacy_models"] == ["en_core_web_sm"]


# ---------------------------------------------------------------------------
# AIGUARD_* environment variables (renamed from LLMGUARD_*)
# ---------------------------------------------------------------------------


def test_aiguard_salt_env(monkeypatch):
    monkeypatch.setenv("AIGUARD_SALT", "env-salt-xyz")
    guard = AIGuard()
    assert guard._config["salt"] == "env-salt-xyz"

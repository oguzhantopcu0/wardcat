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
# The library does not read environment variables (that is the CLI's job)
# ---------------------------------------------------------------------------


def test_library_ignores_aiguard_env(monkeypatch):
    monkeypatch.setenv("AIGUARD_SALT", "env-salt-xyz")
    monkeypatch.setenv("AIGUARD_SPACY_MODEL", "tr_core_news_sm")
    guard = AIGuard()
    assert guard._config["salt"] == ""  # env ignored — pass salt=... explicitly
    assert guard._config["use_ner"] is False  # AIGUARD_SPACY_MODEL ignored too


# ---------------------------------------------------------------------------
# Backend enum (typo-proof LLM backend selection)
# ---------------------------------------------------------------------------


def test_backend_enum_is_str_value():
    from ai_guard import Backend

    assert Backend.OLLAMA == "ollama"
    assert Backend.OPENAI_COMPATIBLE.value == "openai_compatible"
    assert Backend.TRANSFORMERS == "transformers"


def test_backend_enum_configures_llm():
    from ai_guard import AIGuard, Backend

    guard = AIGuard(
        salt="s",
        use_llm=True,
        llm_backend=Backend.OPENAI_COMPATIBLE,
        llm_base_url="http://localhost:8000/v1",
        llm_model="mistral",
    )
    # stored as the canonical string, not the enum object
    assert guard._config["llm_detector"]["backend"] == "openai_compatible"


def test_backend_string_and_enum_equivalent():
    from ai_guard import AIGuard, Backend

    g_enum = AIGuard(salt="s", use_llm=True, llm_backend=Backend.TRANSFORMERS, llm_model="x")
    g_str = AIGuard(salt="s", use_llm=True, llm_backend="transformers", llm_model="x")
    assert (
        g_enum._config["llm_detector"]["backend"] == g_str._config["llm_detector"]["backend"]
    )


def test_invalid_backend_raises():
    from ai_guard import AIGuard
    from ai_guard.exceptions import ConfigError

    with pytest.raises(ConfigError, match="backend"):
        AIGuard(salt="s", use_llm=True, llm_backend="bogus")


# ---------------------------------------------------------------------------
# Fluent builders: with_ner() / with_llm() (chainable, symmetric)
# ---------------------------------------------------------------------------


def test_with_ner_enables_ner():
    from ai_guard import AIGuard, Language

    guard = AIGuard(salt="s").with_ner(language=Language.EN)
    assert guard._config["use_ner"] is True
    assert guard._config["spacy_models"] == ["en_core_web_sm"]


def test_with_ner_requires_model():
    from ai_guard import AIGuard
    from ai_guard.exceptions import ConfigError

    with pytest.raises(ConfigError, match="requires a model"):
        AIGuard(salt="s").with_ner()


def test_with_llm_enables_llm():
    from ai_guard import AIGuard, Backend

    guard = AIGuard(salt="s").with_llm(backend=Backend.OLLAMA, model="llama3.2", adjudicate=True)
    cfg = guard._config["llm_detector"]
    assert cfg["enabled"] is True
    assert cfg["backend"] == "ollama"
    assert cfg["adjudicate"] is True


def test_builders_chain_back_to_back():
    from ai_guard import AIGuard, Backend, Entity, Language

    guard = (
        AIGuard(salt="s")
        .add_entity(Entity.EMAIL, "hash")
        .with_ner(language=Language.EN)
        .with_llm(backend=Backend.OLLAMA, model="llama3.2")
    )
    assert guard._config["use_ner"] is True
    assert guard._config["llm_detector"]["enabled"] is True
    assert len(guard._detectors) == 3  # regex + ner + llm


def test_with_ner_multiple_models():
    from ai_guard import AIGuard

    guard = AIGuard(salt="s").with_ner(
        spacy_model=["en_core_web_sm", "de_core_news_sm"], auto_download=False
    )
    assert guard._config["spacy_models"] == ["en_core_web_sm", "de_core_news_sm"]

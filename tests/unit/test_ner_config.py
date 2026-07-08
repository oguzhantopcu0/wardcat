"""Tests for explicit NER model selection (no default model; Language enum; errors)."""

import pytest

from wardcat import Language, Wardcat
from wardcat.exceptions import ConfigError, UnsupportedLanguageError

# ---------------------------------------------------------------------------
# NER is off by default; no default model is shipped
# ---------------------------------------------------------------------------


def test_bare_guard_has_ner_off():
    guard = Wardcat(salt="s")
    assert guard._config["use_ner"] is False
    assert not guard._config.get("spacy_models")


def test_yaml_use_ner_without_model_raises(tmp_path):
    # A YAML config may switch NER on, but must then name a model (no default).
    import yaml

    cfg_file = tmp_path / "policy.yaml"
    cfg_file.write_text(yaml.dump({"use_ner": True}))
    with pytest.raises(ConfigError, match="no SpaCy model"):
        Wardcat(config_path=str(cfg_file))


def test_with_ner_language_resolves_model_and_enables():
    # with_ner resolves the language to a model and turns NER on.
    guard = Wardcat(salt="s").with_ner(language="de", spacy_size="md", auto_download=False)
    assert guard._config["use_ner"] is True
    assert guard._config["spacy_models"] == ["de_core_news_md"]


# ---------------------------------------------------------------------------
# Language enum (documented selection) — equivalent to the bare code
# ---------------------------------------------------------------------------


def test_language_enum_is_iso_code():
    assert Language.EN == "en"
    assert Language.DE.value == "de"


def test_language_enum_resolves_like_string():
    g_enum = Wardcat(salt="s").with_ner(language=Language.DE)
    g_str = Wardcat(salt="s").with_ner(language="de")
    assert g_enum._config["spacy_models"] == g_str._config["spacy_models"]


def test_language_list_with_enums_multilingual():
    guard = Wardcat(salt="s").with_ner(language=[Language.DE, Language.FR])
    assert guard._config["spacy_models"] == ["de_core_news_sm", "fr_core_news_sm"]


def test_unsupported_language_raises():
    with pytest.raises(UnsupportedLanguageError):
        Wardcat(salt="s").with_ner(language="zz")


# ---------------------------------------------------------------------------
# Explicit spacy_model: single, multiple, implies NER on
# ---------------------------------------------------------------------------


def test_explicit_model_implies_ner_on():
    guard = Wardcat(salt="s").with_ner(spacy_model="en_core_web_sm", auto_download=False)
    assert guard._config["use_ner"] is True
    assert guard._config["spacy_models"] == ["en_core_web_sm"]


def test_multiple_explicit_models():
    guard = Wardcat(salt="s").with_ner(
        spacy_model=["en_core_web_sm", "de_core_news_sm"],
        auto_download=False,
    )
    assert guard._config["spacy_models"] == ["en_core_web_sm", "de_core_news_sm"]


def test_explicit_models_deduped():
    guard = Wardcat(salt="s").with_ner(
        spacy_model=["en_core_web_sm", "en_core_web_sm"],
        auto_download=False,
    )
    assert guard._config["spacy_models"] == ["en_core_web_sm"]


# ---------------------------------------------------------------------------
# The library does not read environment variables (that is the CLI's job)
# ---------------------------------------------------------------------------


def test_library_ignores_aiguard_env(monkeypatch):
    monkeypatch.setenv("WARDCAT_SALT", "env-salt-xyz")
    monkeypatch.setenv("WARDCAT_SPACY_MODEL", "tr_core_news_sm")
    guard = Wardcat()
    assert guard._config["salt"] == ""  # env ignored — pass salt=... explicitly
    assert guard._config["use_ner"] is False  # WARDCAT_SPACY_MODEL ignored too


# ---------------------------------------------------------------------------
# Backend enum (typo-proof LLM backend selection)
# ---------------------------------------------------------------------------


def test_backend_enum_is_str_value():
    from wardcat import Backend

    assert Backend.OLLAMA == "ollama"
    assert Backend.OPENAI_COMPATIBLE.value == "openai_compatible"
    assert Backend.TRANSFORMERS == "transformers"


def test_backend_enum_configures_llm():
    from wardcat import Backend, Wardcat

    guard = Wardcat(salt="s").with_llm(
        backend=Backend.OPENAI_COMPATIBLE,
        base_url="http://localhost:8000/v1",
        model="mistral",
    )
    # stored as the canonical string, not the enum object
    assert guard._config["llm_detector"]["backend"] == "openai_compatible"


def test_backend_string_and_enum_equivalent():
    from wardcat import Backend, Wardcat

    g_enum = Wardcat(salt="s").with_llm(backend=Backend.TRANSFORMERS, model="x")
    g_str = Wardcat(salt="s").with_llm(backend="transformers", model="x")
    assert g_enum._config["llm_detector"]["backend"] == g_str._config["llm_detector"]["backend"]


def test_invalid_backend_raises():
    from wardcat import Wardcat
    from wardcat.exceptions import ConfigError

    with pytest.raises(ConfigError, match="backend"):
        Wardcat(salt="s").with_llm(backend="bogus")


# ---------------------------------------------------------------------------
# Fluent builders: with_ner() / with_llm() (chainable, symmetric)
# ---------------------------------------------------------------------------


def test_with_ner_enables_ner():
    from wardcat import Language, Wardcat

    guard = Wardcat(salt="s").with_ner(language=Language.EN)
    assert guard._config["use_ner"] is True
    assert guard._config["spacy_models"] == ["en_core_web_sm"]


def test_with_ner_requires_model():
    from wardcat import Wardcat
    from wardcat.exceptions import ConfigError

    with pytest.raises(ConfigError, match="requires a model"):
        Wardcat(salt="s").with_ner()


def test_with_llm_enables_llm():
    from wardcat import Backend, Wardcat

    guard = Wardcat(salt="s").with_llm(backend=Backend.OLLAMA, model="llama3.2", adjudicate=True)
    cfg = guard._config["llm_detector"]
    assert cfg["enabled"] is True
    assert cfg["backend"] == "ollama"
    assert cfg["adjudicate"] is True


def test_builders_chain_back_to_back():
    from wardcat import Backend, Entity, Language, Wardcat

    guard = (
        Wardcat(salt="s")
        .add_entity(Entity.EMAIL, "hash")  # regex layer
        .add_entity(Entity.PERSON, "hash")  # NER layer (NER needs an NER entity on)
        .with_ner(language=Language.EN)
        .with_llm(backend=Backend.OLLAMA, model="llama3.2")
    )
    assert guard._config["use_ner"] is True
    assert guard._config["llm_detector"]["enabled"] is True
    assert len(guard._detectors) == 3  # regex + ner + llm


def test_with_ner_multiple_models():
    from wardcat import Wardcat

    guard = Wardcat(salt="s").with_ner(
        spacy_model=["en_core_web_sm", "de_core_news_sm"], auto_download=False
    )
    assert guard._config["spacy_models"] == ["en_core_web_sm", "de_core_news_sm"]

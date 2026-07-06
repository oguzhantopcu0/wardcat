"""Tests for spacy_catalog.py — catalog lookup functions."""

from __future__ import annotations

import pytest

from wardcat.ner.spacy_catalog import (
    SPACY_CATALOG,
    SpacyModelInfo,
    get_models_by_language,
    get_spacy_model,
    recommended_for_language,
    resolve_model,
    supported_languages,
)


class TestSpacyCatalog:
    def test_catalog_is_nonempty(self):
        assert len(SPACY_CATALOG) > 0

    def test_catalog_contains_english_models(self):
        names = {m.name for m in SPACY_CATALOG}
        assert "en_core_web_sm" in names
        assert "en_core_web_md" in names
        assert "en_core_web_lg" in names
        assert "en_core_web_trf" in names

    def test_catalog_contains_turkish_models(self):
        names = {m.name for m in SPACY_CATALOG}
        assert "tr_core_news_md" in names
        assert "tr_core_news_lg" in names

    def test_all_models_have_required_fields(self):
        for m in SPACY_CATALOG:
            assert m.name, f"Model has empty name: {m!r}"
            assert m.language, f"Model {m.name} has empty language"
            assert m.lang_code, f"Model {m.name} has empty lang_code"
            assert m.size in ("sm", "md", "lg", "trf"), (
                f"Model {m.name} has unexpected size: {m.size!r}"
            )
            assert m.ram_mb > 0, f"Model {m.name} has non-positive ram_mb"
            assert m.description, f"Model {m.name} has empty description"

    def test_catalog_covers_all_supported_languages(self):
        lang_codes = {m.lang_code for m in SPACY_CATALOG}
        expected = {"en", "tr", "de", "fr", "es", "it", "nl", "pt"}
        assert expected == lang_codes

    def test_incompatible_model_flagged(self):
        incompatible = [m for m in SPACY_CATALOG if m.incompatible]
        # Turkish trf is the known incompatible model
        assert any(m.name == "tr_core_news_trf" for m in incompatible)

    def test_incompatible_model_has_note(self):
        for m in SPACY_CATALOG:
            if m.incompatible:
                assert m.note, f"Incompatible model {m.name} should have a note"

    def test_wheel_url_models_have_spacy_compat(self):
        for m in SPACY_CATALOG:
            if m.wheel_url:
                assert m.spacy_compat, f"Model {m.name} has wheel_url but no spacy_compat"

    def test_at_least_one_recommended_per_language(self):
        lang_codes = {m.lang_code for m in SPACY_CATALOG}
        for lang in lang_codes:
            rec = recommended_for_language(lang)
            # Every language should have a recommended model
            assert rec is not None, f"No recommended model for language: {lang!r}"

    def test_spacy_model_info_is_frozen(self):
        m = SPACY_CATALOG[0]
        with pytest.raises((AttributeError, TypeError)):
            m.name = "changed"  # type: ignore[misc]


class TestGetSpacyModel:
    def test_found_en_core_web_sm(self):
        m = get_spacy_model("en_core_web_sm")
        assert m is not None
        assert m.name == "en_core_web_sm"
        assert m.language == "English"
        assert m.lang_code == "en"
        assert m.size == "sm"
        assert m.recommended is True

    def test_found_tr_core_news_md(self):
        m = get_spacy_model("tr_core_news_md")
        assert m is not None
        assert m.language == "Turkish"
        assert m.lang_code == "tr"
        assert m.recommended is True
        assert m.wheel_url != ""

    def test_found_incompatible_model(self):
        m = get_spacy_model("tr_core_news_trf")
        assert m is not None
        assert m.incompatible is True
        assert m.note != ""

    def test_not_found_returns_none(self):
        assert get_spacy_model("nonexistent_xyz_model") is None

    def test_not_found_empty_string(self):
        assert get_spacy_model("") is None

    def test_case_sensitive(self):
        # Model names are case-sensitive
        assert get_spacy_model("EN_CORE_WEB_SM") is None

    def test_all_catalog_names_are_retrievable(self):
        for m in SPACY_CATALOG:
            found = get_spacy_model(m.name)
            assert found is not None
            assert found.name == m.name

    def test_returns_correct_size(self):
        for size in ("sm", "md", "lg"):
            m = get_spacy_model(f"en_core_web_{size}")
            if m:
                assert m.size == size


class TestGetModelsByLanguage:
    def test_english_returns_four_models(self):
        models = get_models_by_language("en")
        assert len(models) == 4
        assert all(m.lang_code == "en" for m in models)

    def test_turkish_returns_three_models(self):
        models = get_models_by_language("tr")
        assert len(models) == 3
        assert all(m.lang_code == "tr" for m in models)

    def test_german_returns_models(self):
        models = get_models_by_language("de")
        assert len(models) >= 3
        assert all(m.lang_code == "de" for m in models)

    def test_french_returns_models(self):
        models = get_models_by_language("fr")
        assert len(models) >= 3

    def test_unknown_language_returns_empty_list(self):
        assert get_models_by_language("xx") == []

    def test_empty_string_returns_empty_list(self):
        assert get_models_by_language("") == []

    def test_returns_list_type(self):
        result = get_models_by_language("en")
        assert isinstance(result, list)

    def test_all_models_have_correct_lang_code(self):
        for lang in ("en", "tr", "de", "fr", "es", "it", "nl", "pt"):
            models = get_models_by_language(lang)
            for m in models:
                assert m.lang_code == lang

    def test_returned_models_are_spacy_model_info(self):
        models = get_models_by_language("en")
        for m in models:
            assert isinstance(m, SpacyModelInfo)


class TestRecommendedForLanguage:
    def test_english_recommended_is_sm(self):
        m = recommended_for_language("en")
        assert m is not None
        assert m.name == "en_core_web_sm"
        assert m.recommended is True

    def test_turkish_recommended_is_md(self):
        m = recommended_for_language("tr")
        assert m is not None
        assert m.name == "tr_core_news_md"
        assert m.recommended is True

    def test_german_has_recommended(self):
        m = recommended_for_language("de")
        assert m is not None
        assert m.lang_code == "de"
        assert m.recommended is True

    def test_unknown_language_returns_none(self):
        assert recommended_for_language("xx") is None

    def test_empty_string_returns_none(self):
        assert recommended_for_language("") is None

    def test_returns_spacy_model_info(self):
        m = recommended_for_language("en")
        assert isinstance(m, SpacyModelInfo)

    def test_recommended_model_is_in_catalog(self):
        for lang in ("en", "tr", "de", "fr"):
            m = recommended_for_language(lang)
            if m:
                assert m in SPACY_CATALOG

    def test_at_most_one_recommended_per_language(self):
        lang_codes = {m.lang_code for m in SPACY_CATALOG}
        for lang in lang_codes:
            recommended_models = [m for m in SPACY_CATALOG if m.lang_code == lang and m.recommended]
            assert len(recommended_models) <= 1, (
                f"Language {lang!r} has multiple recommended models: "
                f"{[m.name for m in recommended_models]}"
            )


class TestResolveModel:
    def test_exact_size_match(self):
        assert resolve_model("en", "sm").name == "en_core_web_sm"
        assert resolve_model("de", "md").name == "de_core_news_md"
        assert resolve_model("fr", "lg").name == "fr_core_news_lg"

    def test_default_size_is_sm(self):
        assert resolve_model("en").name == "en_core_web_sm"

    def test_falls_back_to_recommended_when_size_missing(self):
        # Turkish has no "sm" model → falls back to the recommended (md)
        m = resolve_model("tr", "sm")
        assert m is not None
        assert m.lang_code == "tr"
        assert m.recommended

    def test_never_returns_incompatible_model(self):
        # tr_core_news_trf is marked incompatible → must fall back to a usable one
        m = resolve_model("tr", "trf")
        assert m is not None
        assert not m.incompatible

    def test_unsupported_language_returns_none(self):
        assert resolve_model("xx", "sm") is None

    def test_returns_catalog_member(self):
        assert resolve_model("es", "md") in SPACY_CATALOG


class TestSupportedLanguages:
    def test_returns_sorted_unique_codes(self):
        langs = supported_languages()
        assert langs == sorted(set(langs))  # sorted, de-duplicated

    def test_matches_catalog_language_codes(self):
        assert set(supported_languages()) == {m.lang_code for m in SPACY_CATALOG}

    def test_every_supported_language_resolves_to_a_model(self):
        # The detect-then-select contract: a code reported as supported must
        # actually resolve to a usable (compatible) model.
        for code in supported_languages():
            m = resolve_model(code)
            assert m is not None and not m.incompatible

    def test_is_exported_from_package_root(self):
        import wardcat

        assert wardcat.supported_languages() == supported_languages()

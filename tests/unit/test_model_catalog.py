"""
Model catalog unit tests.
"""

from __future__ import annotations

from wardcat.llm.model_catalog import (
    CATALOG,
    DEFAULT_MODEL,
    ModelInfo,
    get_model,
    recommended,
)


class TestCatalogStructure:
    def test_catalog_not_empty(self):
        assert len(CATALOG) > 0

    def test_all_entries_are_model_info(self):
        for m in CATALOG:
            assert isinstance(m, ModelInfo)

    def test_all_have_name_and_description(self):
        for m in CATALOG:
            assert m.name.strip()
            assert m.description.strip()

    def test_all_vram_positive(self):
        for m in CATALOG:
            assert m.vram_gb > 0

    def test_exactly_one_recommended(self):
        recs = [m for m in CATALOG if m.recommended]
        assert len(recs) == 1

    def test_llama31_8b_in_catalog(self):
        names = [m.name for m in CATALOG]
        assert "llama3.1:8b" in names

    def test_default_model_in_catalog(self):
        names = [m.name for m in CATALOG]
        assert DEFAULT_MODEL in names


class TestGetModel:
    def test_known_model_returned(self):
        m = get_model("llama3.1:8b")
        assert m is not None
        assert m.name == "llama3.1:8b"

    def test_unknown_model_returns_none(self):
        assert get_model("gpt-4-turbo") is None

    def test_exact_name_match(self):
        # "llama3.1" ≠ "llama3.1:8b"
        assert get_model("llama3.1") is None


class TestRecommended:
    def test_returns_model_info(self):
        m = recommended()
        assert isinstance(m, ModelInfo)

    def test_recommended_flag_set(self):
        assert recommended().recommended is True

    def test_recommended_is_llama31_8b(self):
        assert recommended().name == "llama3.1:8b"

    def test_recommended_vram_fits_gtx1070(self):
        """The recommended model should fit within the GTX 1070's 8 GB VRAM."""
        assert recommended().vram_gb < 8.0

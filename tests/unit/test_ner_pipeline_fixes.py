"""Parser-only SpaCy pipelines are never used for NER, and NER runs only what it needs.

The German, French and Spanish ``dep_news_trf`` pipelines have no ``ner``
component. Until 1.2.2 ``with_ner(language="de", spacy_size="trf")`` resolved to
one of them, loaded it cleanly and found no names, with nothing on the result to
say so. Resolution now skips them, a warning names the substitute, and loading a
pipeline without NER is refused.

For catalog models whose NER dependencies were measured, the other components
are disabled: the same entities, in a fraction of the time.
"""

from __future__ import annotations

import importlib.util

import pytest

from wardcat import Wardcat
from wardcat.detectors import ner_detector
from wardcat.exceptions import ConfigError
from wardcat.ner.spacy_catalog import (
    SPACY_CATALOG,
    get_spacy_model,
    no_ner_substitute,
    resolve_model,
)

PARSER_ONLY = ("de_dep_news_trf", "fr_dep_news_trf", "es_dep_news_trf")


class TestCatalog:
    @pytest.mark.parametrize("name", PARSER_ONLY)
    def test_parser_only_models_are_marked(self, name: str) -> None:
        info = get_spacy_model(name)
        assert info is not None and info.has_ner is False

    @pytest.mark.parametrize("lang", ["de", "fr", "es"])
    def test_trf_resolves_to_the_large_model(self, lang: str) -> None:
        info = resolve_model(lang, "trf")
        assert info is not None
        assert info.has_ner and info.size == "lg" and info.lang_code == lang

    def test_resolution_never_returns_a_model_without_ner(self) -> None:
        langs = {m.lang_code for m in SPACY_CATALOG}
        for lang in langs:
            for size in ("sm", "md", "lg", "trf"):
                info = resolve_model(lang, size)
                assert info is None or info.has_ner, (lang, size)

    def test_english_trf_is_untouched(self) -> None:
        assert resolve_model("en", "trf").name == "en_core_web_trf"
        assert no_ner_substitute("en", "trf") is None

    def test_measured_pipes_always_include_ner(self) -> None:
        for info in SPACY_CATALOG:
            if info.ner_pipes:
                assert "ner" in info.ner_pipes, info.name


class TestTheSubstitutionIsReported:
    def test_a_note_names_both_models(self) -> None:
        notes = Wardcat._parser_only_notes("de", "trf")
        assert notes == [
            "SpaCy model 'de_dep_news_trf' has no NER component, so the NER layer "
            "is using 'de_core_news_lg' for 'de' instead."
        ]

    def test_no_note_when_nothing_was_swapped(self) -> None:
        assert Wardcat._parser_only_notes(["en", "tr"], "md") == []

    def test_the_note_reaches_every_result(self, monkeypatch) -> None:
        loaded: list[str] = []

        class FakeNER(ner_detector.NERDetector):
            def __init__(self, enabled, model="x"):
                loaded.append(model)
                self.enabled_entities = enabled
                self.nlp = None

            def detect(self, text, candidates=None):
                return []

        monkeypatch.setattr(ner_detector, "NERDetector", FakeNER)
        monkeypatch.setattr("wardcat.guard._resolve_spacy_model", lambda m: m)
        guard = (
            Wardcat(salt="s")
            .add_entity("PERSON")
            .with_ner(language="de", spacy_size="trf", auto_download=False)
        )
        assert loaded == ["de_core_news_lg"]
        assert "has no NER component" in guard.scan("Hallo Welt").warnings[0]


class TestLoading:
    def test_a_pipeline_without_ner_is_refused(self, monkeypatch) -> None:
        spacy = pytest.importorskip("spacy")
        monkeypatch.setattr(spacy, "load", lambda name: spacy.blank("de"))
        monkeypatch.delitem(ner_detector._MODEL_CACHE, "de_dep_news_trf", raising=False)
        with pytest.raises(ConfigError, match="has no NER component"):
            ner_detector._load_model("de_dep_news_trf")
        assert "de_dep_news_trf" not in ner_detector._MODEL_CACHE

    def test_an_unmeasured_pipeline_runs_in_full(self, monkeypatch) -> None:
        spacy = pytest.importorskip("spacy")
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        nlp.add_pipe("ner")
        ner_detector._disable_unused_pipes(nlp, "not_in_the_catalog")
        assert nlp.pipe_names == ["sentencizer", "ner"]

    def test_a_pipeline_unlike_the_measured_one_runs_in_full(self) -> None:
        spacy = pytest.importorskip("spacy")
        nlp = spacy.blank("tr")
        nlp.add_pipe("ner")  # tr_core_news_md is measured with tok2vec + parser
        ner_detector._disable_unused_pipes(nlp, "tr_core_news_md")
        assert nlp.pipe_names == ["ner"]


SAMPLES = {
    "en": "John Smith of Acme Corp met Mary Johnson in London on Monday about the "
    "Berlin office. Dr. Alan Turing wrote to Microsoft and the BBC from Manchester.",
    "tr": "Ahmet Yılmaz dün İstanbul'da Türk Telekom yöneticisi Ayşe Kaya ile görüştü. "
    "Mehmet Demir Ankara'dan Koç Holding adına yazdı; toplantı Kadıköy'de yapılacak.",
}


@pytest.mark.ner
@pytest.mark.parametrize("info", [m for m in SPACY_CATALOG if m.ner_pipes], ids=lambda m: m.name)
def test_measured_pipes_give_the_same_entities(info) -> None:
    if importlib.util.find_spec(info.name) is None:
        pytest.skip(f"{info.name} is not installed")
    spacy = pytest.importorskip("spacy")
    text = SAMPLES.get(info.lang_code)
    if text is None:
        pytest.skip(f"no sample text for {info.lang_code}")
    full = spacy.load(info.name)
    slim = spacy.load(info.name)
    ner_detector._disable_unused_pipes(slim, info.name)
    assert slim.pipe_names == list(info.ner_pipes)

    def ents(nlp):
        return [(e.text, e.label_, e.start_char) for e in nlp(text).ents]

    assert ents(slim) == ents(full)

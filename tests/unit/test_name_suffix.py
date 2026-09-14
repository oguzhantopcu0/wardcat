"""A name keeps one placeholder whatever grammatical case it is written in.

Turkish writes case endings on proper nouns after an apostrophe, and NER models put
the ending inside the entity: "Ahmet Yılmaz'ın" came back as the value, so the same
person hashed differently in every sentence. An index built from those chunks
could not tell the mentions apart. The ending now stays in the text and only the
name is replaced.

None of these tests need SpaCy: model output is supplied directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from wardcat import Action, Entity, Wardcat
from wardcat.detectors.llm_detector import LLMDetector
from wardcat.detectors.ner_detector import NERDetector
from wardcat.utils.text import strip_name_suffix


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Ahmet Yılmaz'ın", "Ahmet Yılmaz"),
        ("Ahmet Yılmaz’a", "Ahmet Yılmaz"),  # typographic apostrophe
        ("Ayşe Demir'den", "Ayşe Demir"),
        ("Mehmet Öztürk'le", "Mehmet Öztürk"),
        ("Ali'nin", "Ali"),
        ("İstanbul'da", "İstanbul"),
        ("Garanti Bankası'nın", "Garanti Bankası"),
        ("Türk Telekom'un", "Türk Telekom"),
        ("Koç Holding'lerinden", "Koç Holding"),
        ("O'Brien'ın", "O'Brien"),  # only the last apostrophe is a suffix
        ("John's", "John"),  # English possessive
        ("ahmet yılmaz'ın", "ahmet yılmaz"),  # a lower-cased document
        ("Ahmet Yılmaz’a e", "Ahmet Yılmaz"),  # the model swallowed "e" of "e-posta"
    ],
)
def test_suffix_is_cut(value, expected):
    start = 10
    trimmed, new_start, new_end = strip_name_suffix(value, start, start + len(value))
    assert trimmed == expected
    assert (new_start, new_end) == (start, start + len(expected))


@pytest.mark.parametrize(
    "value",
    [
        "O'Brien",  # the apostrophe belongs to the name
        "D'Angelo",
        "o'brien",  # lower case: one letter before the apostrophe is not a name to cut
        "Rock'n",  # no vowel in what follows
        "Ahmet Yılmaz",
        "Ahmet Yılmaz'",  # nothing after the apostrophe
        "Ahmet Yılmaz'ıncelemedeydiklerinden",  # too long to be an ending
        "5'in",  # digits are not a name
    ],
)
def test_names_without_a_suffix_are_left_alone(value):
    assert strip_name_suffix(value, 0, len(value)) == (value, 0, len(value))


def _ner_guard(monkeypatch, ents: list[tuple[str, str, int, int]], action=Action.HASH) -> Wardcat:
    doc = MagicMock(
        ents=[
            MagicMock(label_=label, text=text, start_char=start, end_char=end)
            for text, label, start, end in ents
        ]
    )
    nlp = MagicMock(return_value=doc)
    monkeypatch.setattr("wardcat.detectors.ner_detector._load_model", lambda name: nlp)
    monkeypatch.setattr("wardcat.guard._resolve_spacy_model", lambda name: name)
    return (
        Wardcat(salt="s")
        .with_ner(spacy_model="tr_core_news_md", auto_download=False)
        .add_entities([Entity.PERSON, Entity.LOCATION], action=action)
    )


def _ent(text: str, value: str, label: str) -> tuple[str, str, int, int]:
    start = text.index(value)
    return value, label, start, start + len(value)


def test_one_person_gets_one_hash_in_every_case(monkeypatch):
    texts = [
        "Müşteri Ahmet Yılmaz iade istedi.",
        "Ahmet Yılmaz'ın talebi değerlendirildi.",
        "Ahmet Yılmaz’a e-posta gönderildi.",
    ]
    names = ["Ahmet Yılmaz", "Ahmet Yılmaz'ın", "Ahmet Yılmaz’a e"]
    sanitized = []
    for text, name in zip(texts, names, strict=True):
        guard = _ner_guard(monkeypatch, [_ent(text, name, "PERSON")])
        result = guard.scan(text)
        assert result.violations[0].original == "Ahmet Yılmaz"
        sanitized.append(result.sanitized_text)

    token = sanitized[0].split()[1]
    assert sanitized == [
        f"Müşteri {token} iade istedi.",
        f"{token}'ın talebi değerlendirildi.",
        f"{token}’a e-posta gönderildi.",
    ]


def test_the_ending_stays_readable_with_redact(monkeypatch):
    text = "Toplantı İstanbul'da, Ayşe Demir'den haber bekleniyor."
    guard = _ner_guard(
        monkeypatch,
        [_ent(text, "İstanbul'da", "GPE"), _ent(text, "Ayşe Demir'den", "PERSON")],
        action=Action.REDACT,
    )
    assert (
        guard.scan(text).sanitized_text == "Toplantı [LOCATION]'da, [PERSON]'den haber bekleniyor."
    )


def test_ner_span_offsets_point_at_the_name(monkeypatch):
    text = "Dosyayı Ali Veli'nin masasına bıraktım."
    doc = MagicMock(ents=[MagicMock(label_="PER", text="Ali Veli'nin", start_char=8, end_char=20)])
    monkeypatch.setattr(
        "wardcat.detectors.ner_detector._load_model", lambda n: MagicMock(return_value=doc)
    )
    (span,) = NERDetector({"PERSON"}, model="tr_core_news_md").detect(text)
    assert (span.text, text[span.start : span.end]) == ("Ali Veli", "Ali Veli")


def test_llm_quoting_the_suffixed_form_locates_every_mention():
    backend = MagicMock()
    backend.complete_messages.return_value = '[{"type":"PERSON","text":"Ahmet Yılmaz\'ın"}]'
    detector = LLMDetector(backend, {"PERSON"})
    text = "Ahmet Yılmaz aradı. Ahmet Yılmaz'ın dosyası açık."
    spans = detector.detect(text)
    assert [(s.start, s.end) for s in spans] == [(0, 12), (20, 32)]
    assert {text[s.start : s.end] for s in spans} == {"Ahmet Yılmaz"}

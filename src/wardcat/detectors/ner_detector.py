from __future__ import annotations

import logging
import re
import threading
from typing import Any

from wardcat.detectors.base import BaseDetector, DetectedSpan
from wardcat.utils.logsafe import describe
from wardcat.utils.text import strip_name_suffix

logger = logging.getLogger(__name__)

# SpaCy label → our entity type mapping
# Includes English (en_core_web_sm) and Turkish (tr_core_news_sm) models
_SPACY_LABEL_MAP: dict[str, str] = {
    # English model labels
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "LOCATION",  # Geopolitical entity — country, city, region
    "LOC": "LOCATION",  # Non-political location — mountain, river, area
    # Turkish model labels (tr_core_news_sm / tr_core_news_md / tr_core_news_lg)
    "PER": "PERSON",  # Person name in Turkish model
    # NORP is nationality, religious or political group — GDPR Article 9 data,
    # not a company. Mapping it to ORG labelled it as one and, with ORG's `warn`
    # action, left it in the text. It gets its own type, off by default: these
    # words ("Turkish", "Catholic") are ordinary vocabulary, so redacting every
    # one of them by default would wreck the text for no gain outside Art. 9 work.
    "NORP": "NRP",
    "FAC": "ADDRESS",  # Building, bridge, etc.
}

# ── Span-level false-positive filters ─────────────────────────────────────────
# NER models mis-label addresses, codes and short tokens as PERSON or ORG.
# These patterns identify spans that are clearly neither.

# Span contains digits or characters typical of addresses/codes (No:, /, :)
_NON_PERSON_CHARS = re.compile(r"[0-9/:]")

# Punctuation a model sometimes swallows at a span's edge — a quote it opened,
# the colon after a speaker's name. Trimming beats rejecting: `tracy:"i'm` is a
# real name wearing punctuation, and dropping the whole span loses the name.
_EDGE_PUNCT = "\"'`:;,.()[]{}<>!?\\ \n\t\r"

# Span contains address-type keywords (Turkish + English + German + French)
_ADDRESS_KW = re.compile(
    r"\b(?:"
    r"Caddesi|Cad\.|Sokağı|Sokak|Sok\.|Mahallesi|Mah\.|Bulvarı|Blv\."
    r"|Apartmanı|Apt\.|Sitesi|No\b"
    r"|Street|St\.|Avenue|Ave\.|Road|Rd\.|Boulevard|Lane|Drive|Court"
    r"|Stra(?:ss|ß)e|Gasse|Weg|Platz|Allee"  # German
    r"|Rue|Avenue|Boulevard|Impasse|Allée|Place"  # French
    r")\b",
    re.IGNORECASE,
)

# ── Multilingual gazetteer: tokens that are never PII on their own ─────────────
# NER models often mis-label job titles, HR terms, and abbreviations as
# PERSON/ORG/ADDRESS (e.g. "Senior Backend Engineer", "New hire", "T.C.").
# A span composed ENTIRELY of these tokens is rejected.
_NER_STOPWORDS: frozenset[str] = frozenset(
    {
        # Job-title / seniority words — EN
        "senior",
        "junior",
        "lead",
        "principal",
        "staff",
        "chief",
        "head",
        "manager",
        "director",
        "engineer",
        "developer",
        "analyst",
        "officer",
        "consultant",
        "specialist",
        "coordinator",
        "administrator",
        "assistant",
        "backend",
        "frontend",
        "fullstack",
        "software",
        "hardware",
        "data",
        "product",
        "project",
        "team",
        "platform",
        "new",
        "hire",
        "candidate",
        "employee",
        "employer",
        "intern",
        "contractor",
        "applicant",
        "onboarding",
        # German
        "ingenieur",
        "entwickler",
        "leiter",
        "geschäftsführer",
        "mitarbeiter",
        "berater",
        "abteilung",
        "neuer",
        "neue",
        "kandidat",
        # French
        "ingénieur",
        "développeur",
        "developpeur",
        "directeur",
        "responsable",
        "chef",
        "employé",
        "employe",
        "stagiaire",
        "candidat",
        "nouveau",
        "nouvel",
        # Turkish
        "müdür",
        "mudur",
        "yönetici",
        "yonetici",
        "uzman",
        "danışman",
        "danisman",
        "mühendis",
        "muhendis",
        "geliştirici",
        "gelistirici",
        "temsilci",
        "müşteri",
        "musteri",
        "çalışan",
        "calisan",
        "aday",
        "yeni",
        "personel",
        # Abbreviations / non-name tokens
        "tc",
        "t.c",
        "t.c.",
        "vkn",
        "mr",
        "mrs",
        "ms",
        "dr",
        "herr",
        "frau",
        # Form-field labels and postal designators. A model reads "SSN", "IBAN"
        # or "APO AP" beside a value as an organisation; they name the field or
        # the military post office, never a company or a person.
        "ssn",
        "iban",
        "bic",
        "swift",
        "cvv",
        "cvc",
        "pin",
        "dob",
        "vat",
        "atm",
        "address",
        "phone",
        "email",
        "e-mail",
        "fax",
        "mobile",
        "tel",
        "adres",
        "telefon",
        "e-posta",
        "suite",
        "apt",
        "p.o",
        "box",
        "apo",
        "fpo",
        "dpo",
        "psc",
        "rr",
        "aa",  # the armed-forces "state" codes that follow APO/FPO/DPO
        "ae",
        "ap",
    }
)

# Words a model sweeps into the front of a name: an article, a greeting, a
# "Sayın" or a "dün" that happened to stand before it. They are trimmed only from
# the front and only while they lead, so "de la Cruz" or "van der Berg" — whose
# particles are not listed — stay whole. Anything uncertain stays in the span:
# a word too many costs a consistent placeholder, a word too few leaks a name.
# That is why "a", "an" and "cher" are absent: "A Smith", "An Nguyen", "Cher".
_LEADING_NOISE: frozenset[str] = frozenset(
    {
        "the",
        "dear",
        "hi",
        "hello",
        "hey",
        "attn",
        "cc",
        "sayın",
        "sayin",
        "merhaba",
        "selam",
        "dün",
        "dun",
        "bugün",
        "bugun",
        "yarın",
        "yarin",
        "liebe",
        "lieber",
        "sehr",
        "geehrte",
        "geehrter",
        "bonjour",
    }
)

# A street designator as the final word makes an "organisation" a street name —
# "Pollen Crescent", "Koepenicker Str". Final word only: "Wall Street Journal"
# keeps its place.
_STREET_LAST_WORD = re.compile(
    r"(?:street|st|str|straße|strasse|avenue|ave|road|rd|boulevard|blvd|lane|drive"
    r"|crescent|caddesi|cad|sokak|sokağı|sok|mahallesi|mah|bulvarı|gasse|weg|allee)\.?",
    re.IGNORECASE,
)


def _trim_span(text: str, start: int, end: int) -> tuple[str, int, int]:
    """Strip punctuation and whitespace from a span's edges, adjusting offsets."""
    trimmed = text.strip(_EDGE_PUNCT)
    if not trimmed:
        return text, start, end
    offset = text.index(trimmed)
    return trimmed, start + offset, start + offset + len(trimmed)


# A legal form at the end of a line marks the line that names an organisation.
_LEGAL_FORM = re.compile(
    r"\b(?:Inc|Ltd|LLC|PLC|Corp|Co|GmbH|AG|KG|SA|SAS|SARL|A\.Ş|Ltd\. Şti|Holding|Group|Grup)\.?$",
    re.IGNORECASE,
)


def _best_line(text: str, start: int, end: int) -> tuple[str, int, int]:
    """Keep the one line of a multi-line span that names the entity.

    A name does not continue onto the next line. In an address block or a
    signature the model runs on into the following field — "Anna Josefsen\nAddress"
    — and in a document with headings it runs back over the heading above —
    "Renewals Team\nDalton Inc.". The line with the most capitalised words is
    the name; a line ending in a legal form wins for an organisation; ties go
    to the first line.
    """
    if "\n" not in text and "\r" not in text:
        return text, start, end
    best: tuple[int, int, str] | None = None  # (score, -position, line)
    pos = 0
    for line in re.split(r"\r?\n", text):
        stripped = line.strip(_EDGE_PUNCT)
        if stripped:
            score = sum(1 for w in stripped.split() if w[:1].isupper())
            if _LEGAL_FORM.search(stripped):
                score += 2
            if best is None or score > best[0]:
                best = (score, -pos, line)
        pos += len(line) + 1
    if best is None:
        return text, start, end
    line_start = start - best[1]
    return _trim_span(best[2], line_start, line_start + len(best[2]))


# Where a model's span runs into markup or a record delimiter — "Carolyn
# Hill</name", "Dawn Perkins|560=726", 'BkCode=1290:::ABC Bank' — the name is one
# fragment between delimiters. A comma counts only before a number, so
# "Marshall, Hernandez and Simpson" stays whole.
_STRUCTURAL_DELIMITER = re.compile(r"""[<>|/\\="`\[\]{}*;:]+|,(?=\s*\d)|\s-\s""")


def _name_fragment(text: str, start: int, end: int) -> tuple[str, int, int]:
    """Keep the longest delimiter-free fragment that carries no digit.

    Only applies when the span holds a structural delimiter; a span with none
    is returned as it is. A span whose every fragment has a digit is returned
    unchanged too, for the digit filter to reject.
    """
    if not _STRUCTURAL_DELIMITER.search(text):
        return text, start, end
    best: tuple[int, int, str] | None = None
    pos = 0
    for piece in _STRUCTURAL_DELIMITER.split(text):
        at = text.find(piece, pos)
        stripped = piece.strip(_EDGE_PUNCT)
        if stripped and not any(c.isdigit() for c in stripped):
            if best is None or len(stripped) > best[0]:
                best = (len(stripped), at, piece)
        pos = at + len(piece)
    if best is None:
        return text, start, end
    frag_start = start + best[1]
    return _trim_span(best[2], frag_start, frag_start + len(best[2]))


def _drop_leading_noise(text: str, start: int, end: int) -> tuple[str, int, int]:
    """Remove :data:`_LEADING_NOISE` words from the front, keeping at least one word."""
    while True:
        head, sep, rest = text.partition(" ")
        if not sep or not rest.strip() or head.strip(".,:;").lower() not in _LEADING_NOISE:
            return text, start, end
        offset = len(head) + len(sep) + (len(rest) - len(rest.lstrip()))
        text, start = text[offset:], start + offset


def _has_case(text: str) -> bool:
    """Does this text use capitalization at all?

    Capitalization is only evidence about a span when the surrounding document
    actually uses it. Chat logs, ASR output and lower-cased pipelines carry none,
    and judging "no capital letter, so not a name" there rejects real names for a
    property the text never had. One uppercase letter in every two hundred is a
    stray acronym, not a document that capitalizes.
    """
    letters = sum(1 for c in text if c.isalpha())
    if not letters:
        return False
    return sum(1 for c in text if c.isupper()) / letters > 0.005


def _is_all_stopwords(text: str) -> bool:
    """Return True if every token in the span is a known non-PII stopword.

    Tokenizes on whitespace, strips surrounding punctuation, lowercases.
    An empty result (only punctuation) counts as stopwords too.
    """
    tokens = [t.strip(".,;:/-()").lower() for t in text.split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return True
    return all(t in _NER_STOPWORDS for t in tokens)


def _is_valid_person(text: str, *, document_has_case: bool = True) -> bool:
    """Return False if the span is unlikely to be a real person name.

    Filters out:
    - Very short tokens (≤2 chars) — abbreviations like "TC", "Mr"
    - Spans containing digits or address punctuation — "No:42", "Blok/3"
    - Spans containing street/address keywords — "Moda Caddesi No:42"
    - Spans with no uppercase-initial word — "adresine veya", "veya", common words

    The last rule holds only where capitalization means something. Pass
    ``document_has_case=False`` for a text that does not capitalize at all (see
    :func:`_has_case`); there the rule rejects real names for a property the
    document never had.
    """
    stripped = text.strip()
    if len(stripped) <= 2:
        logger.debug("NER PERSON filtered (too short): %s", describe(text))
        return False
    if _NON_PERSON_CHARS.search(stripped):
        logger.debug("NER PERSON filtered (contains digits/punct): %s", describe(text))
        return False
    if _ADDRESS_KW.search(stripped):
        logger.debug("NER PERSON filtered (address keyword): %s", describe(text))
        return False
    # At least one word must start with an uppercase letter — but only where the
    # document capitalizes at all.
    if document_has_case and not any(word[:1].isupper() for word in stripped.split()):
        logger.debug("NER PERSON filtered (no uppercase word): %s", describe(text))
        return False
    return True


# ── SpaCy model singleton cache ────────────────────────────────────────────
# The SpaCy nlp object is loaded only once per model name.
# Thread-safe: protected by _CACHE_LOCK.
# Effect: even if multiple Wardcat instances are created, SpaCy is kept
# in memory only once (~300–500 MB savings per instance).
_MODEL_CACHE: dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()


def _load_model(model_name: str) -> Any:
    """Return the SpaCy model from cache; load and cache it if not present."""
    with _CACHE_LOCK:
        if model_name not in _MODEL_CACHE:
            import spacy  # lazy import — SpaCy is optional

            logger.info("Loading SpaCy model: %s", model_name)
            _MODEL_CACHE[model_name] = spacy.load(model_name)
            logger.info("SpaCy model ready: %s", model_name)
        return _MODEL_CACHE[model_name]


class NERDetector(BaseDetector):
    """SpaCy-based Named Entity Recognition detector."""

    layer = "ner"

    def __init__(self, enabled_entities: set[str], model: str = "en_core_web_sm") -> None:
        self.nlp = _load_model(model)
        self.enabled_entities = enabled_entities

    def detect(self, text: str, candidates: list[DetectedSpan] | None = None) -> list[DetectedSpan]:
        """Return person, organization, and location spans detected by SpaCy NER."""
        return self._spans_from_doc(self.nlp(text), text)

    def detect_many(self, texts: list[str]) -> list[list[DetectedSpan]]:
        """One ``nlp.pipe`` pass over the batch — the model's own batching, which
        is several times faster than a call per text."""
        return [
            self._spans_from_doc(doc, text)
            for doc, text in zip(self.nlp.pipe(texts, batch_size=32), texts, strict=True)
        ]

    def _spans_from_doc(self, doc: Any, text: str) -> list[DetectedSpan]:
        document_has_case = _has_case(text)
        spans: list[DetectedSpan] = []
        for ent in doc.ents:
            mapped = _SPACY_LABEL_MAP.get(ent.label_)
            if not mapped or mapped not in self.enabled_entities:
                continue
            value, start, end = _trim_span(ent.text, ent.start_char, ent.end_char)
            value, start, end = _best_line(value, start, end)
            if mapped == "PERSON":
                # Measured on the held-out corpus: recovering a name from markup
                # adds people at little cost, but applied to organisations it
                # keeps field labels ("BkName") as companies — more noise than gain.
                value, start, end = _name_fragment(value, start, end)
            value, start, end = _drop_leading_noise(value, start, end)
            value, start, end = strip_name_suffix(value, start, end)
            # Multilingual gazetteer filter: drop spans that are entirely
            # job titles, HR terms, or abbreviations (never PII on their own).
            if _is_all_stopwords(value):
                logger.debug("NER %s filtered (all stopwords): %s", mapped, describe(value))
                continue
            if mapped == "PERSON" and not _is_valid_person(
                value, document_has_case=document_has_case
            ):
                continue
            # An ORG carrying digits or address punctuation is an address fragment
            # or a code, not a company: models label "CO Uruguay 64677" ORG readily.
            # The street-keyword list is deliberately *not* applied here — it would
            # take "Wall Street Journal" with it.
            if mapped == "ORG" and _NON_PERSON_CHARS.search(value):
                logger.debug("NER ORG filtered (digits/address punctuation): %s", describe(value))
                continue
            if mapped == "ORG" and sum(c.isalpha() for c in value) < 2:
                logger.debug("NER ORG filtered (fewer than two letters): %s", describe(value))
                continue
            if mapped == "ORG" and _STREET_LAST_WORD.fullmatch(value.split()[-1]):
                logger.debug("NER ORG filtered (ends in a street designator): %s", describe(value))
                continue
            spans.append(
                DetectedSpan(
                    entity_type=mapped,
                    text=value,
                    start=start,
                    end=end,
                    confidence=0.85,
                )
            )
        return spans

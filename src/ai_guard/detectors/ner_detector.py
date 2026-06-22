from __future__ import annotations

import logging
import re
import threading
from typing import Any

from ai_guard.detectors.base import BaseDetector, DetectedSpan

logger = logging.getLogger(__name__)

# SpaCy label → our entity type mapping
# Includes English (en_core_web_sm) and Turkish (tr_core_news_sm) models
_SPACY_LABEL_MAP: dict[str, str] = {
    # English model labels
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "ADDRESS",  # Geopolitical entity
    "LOC": "ADDRESS",  # Location
    # Turkish model labels (tr_core_news_sm / tr_core_news_md / tr_core_news_lg)
    "PER": "PERSON",  # Person name in Turkish model
    "NORP": "ORG",  # Nationality, religious group, etc.
    "FAC": "ADDRESS",  # Building, bridge, etc.
}

# ── PERSON false-positive filters ─────────────────────────────────────────────
# Turkish NER models often mis-label addresses and short tokens as PERSON.
# These patterns identify spans that are clearly NOT person names.

# Span contains digits or characters typical of addresses/codes (No:, /, :)
_NON_PERSON_CHARS = re.compile(r"[0-9/:]")

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
    }
)


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


def _is_valid_person(text: str) -> bool:
    """Return False if the span is unlikely to be a real person name.

    Filters out:
    - Very short tokens (≤2 chars) — abbreviations like "TC", "Mr"
    - Spans containing digits or address punctuation — "No:42", "Blok/3"
    - Spans containing street/address keywords — "Moda Caddesi No:42"
    - Spans with no uppercase-initial word — "adresine veya", "veya", common words
    """
    stripped = text.strip()
    if len(stripped) <= 2:
        logger.debug("NER PERSON filtered (too short): %r", text)
        return False
    if _NON_PERSON_CHARS.search(stripped):
        logger.debug("NER PERSON filtered (contains digits/punct): %r", text)
        return False
    if _ADDRESS_KW.search(stripped):
        logger.debug("NER PERSON filtered (address keyword): %r", text)
        return False
    # At least one word must start with an uppercase letter.
    # Person names are always capitalized; common word sequences ("adresine veya") are not.
    if not any(word[:1].isupper() for word in stripped.split()):
        logger.debug("NER PERSON filtered (no uppercase word): %r", text)
        return False
    return True


# ── SpaCy model singleton cache ────────────────────────────────────────────
# The SpaCy nlp object is loaded only once per model name.
# Thread-safe: protected by _CACHE_LOCK.
# Effect: even if multiple AIGuard instances are created, SpaCy is kept
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

    def __init__(self, enabled_entities: set[str], model: str = "en_core_web_sm") -> None:
        self.nlp = _load_model(model)
        self.enabled_entities = enabled_entities

    def detect(
        self, text: str, candidates: list[DetectedSpan] | None = None
    ) -> list[DetectedSpan]:
        """Return person, organization, and location spans detected by SpaCy NER."""
        doc = self.nlp(text)
        spans: list[DetectedSpan] = []
        for ent in doc.ents:
            mapped = _SPACY_LABEL_MAP.get(ent.label_)
            if not mapped or mapped not in self.enabled_entities:
                continue
            # Multilingual gazetteer filter: drop spans that are entirely
            # job titles, HR terms, or abbreviations (never PII on their own).
            if _is_all_stopwords(ent.text):
                logger.debug("NER %s filtered (all stopwords): %r", mapped, ent.text)
                continue
            # Apply false-positive filter for PERSON entities.
            if mapped == "PERSON" and not _is_valid_person(ent.text):
                continue
            spans.append(
                DetectedSpan(
                    entity_type=mapped,
                    text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.85,
                )
            )
        return spans

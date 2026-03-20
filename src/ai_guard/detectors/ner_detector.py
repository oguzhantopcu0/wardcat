from __future__ import annotations

import logging
import re
import threading
from typing import Any, List, Set

from ai_guard.detectors.base import BaseDetector, DetectedSpan

logger = logging.getLogger(__name__)

# SpaCy label → our entity type mapping
# Includes English (en_core_web_sm) and Turkish (tr_core_news_sm) models
_SPACY_LABEL_MAP: dict[str, str] = {
    # English model labels
    "PERSON": "PERSON",
    "ORG":    "ORG",
    "GPE":    "ADDRESS",   # Geopolitical entity
    "LOC":    "ADDRESS",   # Location
    # Turkish model labels (tr_core_news_sm / tr_core_news_md / tr_core_news_lg)
    "PER":    "PERSON",    # Person name in Turkish model
    "NORP":   "ORG",       # Nationality, religious group, etc.
    "FAC":    "ADDRESS",   # Building, bridge, etc.
}

# ── PERSON false-positive filters ─────────────────────────────────────────────
# Turkish NER models often mis-label addresses and short tokens as PERSON.
# These patterns identify spans that are clearly NOT person names.

# Span contains digits or characters typical of addresses/codes (No:, /, :)
_NON_PERSON_CHARS = re.compile(r"[0-9/:]")

# Span contains address-type keywords (Turkish + English)
_ADDRESS_KW = re.compile(
    r"\b(?:"
    r"Caddesi|Cad\.|Sokağı|Sokak|Sok\.|Mahallesi|Mah\.|Bulvarı|Blv\."
    r"|Apartmanı|Apt\.|Sitesi|No\b"
    r"|Street|St\.|Avenue|Ave\.|Road|Rd\.|Boulevard|Lane|Drive|Court"
    r")\b",
    re.IGNORECASE,
)


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
# Effect: even if multiple LLMGuard instances are created, SpaCy is kept
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

    def __init__(self, enabled_entities: Set[str], model: str = "en_core_web_sm") -> None:
        self.nlp = _load_model(model)
        self.enabled_entities = enabled_entities

    def detect(self, text: str) -> List[DetectedSpan]:
        """Return person, organization, and location spans detected by SpaCy NER."""
        doc = self.nlp(text)
        spans: List[DetectedSpan] = []
        for ent in doc.ents:
            mapped = _SPACY_LABEL_MAP.get(ent.label_)
            if not mapped or mapped not in self.enabled_entities:
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
                )
            )
        return spans

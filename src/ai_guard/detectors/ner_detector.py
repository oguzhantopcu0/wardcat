from __future__ import annotations

import logging
import threading
from typing import Any, List, Set

from ai_guard.detectors.base import BaseDetector, DetectedSpan

logger = logging.getLogger(__name__)

# SpaCy etiket → bizim entity tipi eşlemesi
# İngilizce (en_core_web_sm) ve Türkçe (tr_core_news_sm) modelleri dahil
_SPACY_LABEL_MAP: dict[str, str] = {
    # İngilizce model etiketleri
    "PERSON": "PERSON",
    "ORG":    "ORG",
    "GPE":    "ADDRESS",   # Geopolitical entity
    "LOC":    "ADDRESS",   # Location
    # Türkçe model etiketleri (tr_core_news_sm / tr_core_news_md / tr_core_news_lg)
    "PER":    "PERSON",    # tr modelinde kişi adı
    "NORP":   "ORG",       # Milliyet, dini grup vb.
    "FAC":    "ADDRESS",   # Bina, köprü vb.
}

# ── SpaCy model singleton cache ────────────────────────────────────────────
# Her model adı için SpaCy nlp nesnesi yalnızca bir kez yüklenir.
# Thread-safe: _CACHE_LOCK ile korunur.
# Etki: LLMGuard birden fazla instance oluşturulsa dahi SpaCy bellekte
# yalnızca bir kez tutulur (~300–500 MB tasarruf/instance).
_MODEL_CACHE: dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()


def _load_model(model_name: str) -> Any:
    """SpaCy modelini cache'ten döndürür; yoksa yükler ve cache'e ekler."""
    with _CACHE_LOCK:
        if model_name not in _MODEL_CACHE:
            import spacy  # lazy import — SpaCy opsiyoneldir
            logger.info("SpaCy modeli yükleniyor: %s", model_name)
            _MODEL_CACHE[model_name] = spacy.load(model_name)
            logger.info("SpaCy modeli hazır: %s", model_name)
        return _MODEL_CACHE[model_name]


class NERDetector(BaseDetector):
    """SpaCy tabanlı Named Entity Recognition dedektörü."""

    def __init__(self, enabled_entities: Set[str], model: str = "en_core_web_sm") -> None:
        self.nlp = _load_model(model)
        self.enabled_entities = enabled_entities

    def detect(self, text: str) -> List[DetectedSpan]:
        """SpaCy NER ile kişi, kurum ve konum spanlarını döndür."""
        doc = self.nlp(text)
        spans: List[DetectedSpan] = []
        for ent in doc.ents:
            mapped = _SPACY_LABEL_MAP.get(ent.label_)
            if mapped and mapped in self.enabled_entities:
                spans.append(
                    DetectedSpan(
                        entity_type=mapped,
                        text=ent.text,
                        start=ent.start_char,
                        end=ent.end_char,
                    )
                )
        return spans

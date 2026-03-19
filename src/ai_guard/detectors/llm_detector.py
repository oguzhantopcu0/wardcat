"""
LLM tabanlı PII dedektörü.

On-prem Llama (veya başka bir model) üzerinden PII tespiti yapar.
Regex ve NER dedektörleriyle aynı BaseDetector arayüzünü uygular,
dolayısıyla DetectionEngine tarafından şeffaf biçimde kullanılır.

Tasarım kararı: LangChain / LangGraph KULLANILMADI.
Gerekçe: Tek bir prompt → JSON parse → DetectedSpan dönüşümü için
100+ geçişli bağımlılık gereksizdir. Doğrudan httpx ile yapılan
implementasyon daha hafif, daha test edilebilir ve daha şeffaftır.
"""
from __future__ import annotations

import json
import logging
import re
from typing import List, Set

from ai_guard.detectors.base import BaseDetector, DetectedSpan
from ai_guard.llm.backends.base import BaseLLMBackend
from ai_guard.llm.prompt import build_prompt

logger = logging.getLogger(__name__)

# LLM yanıtından JSON array'i çıkarmak için
_JSON_RE = re.compile(r"\[.*?\]", re.DOTALL)

# Yapısal entity'ler için minimum format doğrulama kalıpları.
# LLM bu tipleri döndürürse içerik de formatla örtüşmeli;
# örtüşmezse halüsinasyon olarak atılır.
_STRUCTURAL_VALIDATORS: dict[str, re.Pattern] = {
    # Kişi adı en az iki kelimeden oluşmalı (ad + soyad).
    # Tek kelimeler (ör. "hedef", "müşteri") LLM halüsinasyonudur → atılır.
    "PERSON":     re.compile(r"^\S+(?:\s+\S+)+$"),
    "TC_ID":      re.compile(r"^\d{11}$"),
    "IBAN":       re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9 ]{10,}$", re.IGNORECASE),
    "CREDIT_CARD": re.compile(r"^[\d\s\-]{13,19}$"),
    "PHONE":      re.compile(r"[\d\s\-\+\(\)]{7,}"),
    "IP_ADDRESS": re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$"),
    "POSTAL_CODE": re.compile(r"^\d{5}$"),
}


class LLMDetector(BaseDetector):
    """
    On-prem LLM üzerinden PII tespiti.

    Desteklenen backend'ler:
    - OllamaBackend  — Ollama REST API (yerel model çalıştırma)
    - OpenAICompatBackend — vLLM, LM Studio, LocalAI, LiteLLM

    Hata durumunda (bağlantı kesilmesi, bozuk JSON vb.) WARNING loglar
    ve boş liste döndürür — diğer dedektörlerin çalışması engellenmez.
    """

    def __init__(
        self,
        backend: BaseLLMBackend,
        enabled_entities: Set[str],
        *,
        timeout: int = 60,
    ) -> None:
        self.backend          = backend
        self.enabled_entities = enabled_entities
        self.timeout          = timeout

    def detect(self, text: str) -> List[DetectedSpan]:
        if not text.strip():
            return []

        prompt = build_prompt(text, self.enabled_entities)
        try:
            raw = self.backend.complete(prompt, timeout=self.timeout)
            entities = self._parse_llm_response(raw)
            spans = self._locate_spans(text, entities)
            logger.debug("LLM dedektörü: %d entity döndü, %d span konumlandı", len(entities), len(spans))
            return spans
        except ConnectionError as exc:
            logger.warning("LLM dedektörü bağlantı hatası: %s", exc)
        except Exception as exc:
            logger.warning("LLM dedektörü başarısız: %s", exc, exc_info=True)
        return []

    # ------------------------------------------------------------------

    def _parse_llm_response(self, raw: str) -> list[dict]:
        """
        LLM yanıtından JSON array'i ayıklar.

        Küçük modeller zaman zaman ```json ... ``` bloğu veya
        açıklama metni ekler; regex ile temizlenir.
        """
        # Markdown kod bloğunu soy
        raw = re.sub(r"```(?:json)?", "", raw).strip()

        match = _JSON_RE.search(raw)
        if not match:
            logger.debug("LLM yanıtında JSON array bulunamadı. Ham yanıt: %.200r", raw)
            return []

        try:
            data = json.loads(match.group())
            if not isinstance(data, list):
                logger.debug("LLM JSON yanıtı list değil: %r", type(data).__name__)
                return []
            return data
        except json.JSONDecodeError as exc:
            logger.debug("LLM yanıtı JSON parse hatası: %s — ham: %.200r", exc, raw)
            return []

    def _locate_spans(self, text: str, entities: list[dict]) -> List[DetectedSpan]:
        """
        LLM'in döndürdüğü entity metinlerini orijinal metinde konumlandırır.

        LLM bazen metni hafifçe değiştirir (büyük/küçük harf vb.);
        eşleşme bulunamazsa o entity atlanır.
        """
        spans: List[DetectedSpan] = []
        seen: set[tuple[int, int]] = set()   # tekrar konum kontrolü

        for item in entities:
            entity_type = str(item.get("type", "")).upper().strip()
            entity_text = str(item.get("text", "")).strip()

            if not entity_text or entity_type not in self.enabled_entities:
                continue

            # Yapısal entity'ler için format doğrulaması:
            # LLM yanlış içerik atadıysa (halüsinasyon) sessizce atla.
            validator = _STRUCTURAL_VALIDATORS.get(entity_type)
            if validator and not validator.search(entity_text):
                logger.debug(
                    "Halüsinasyon filtresi: %s %r format doğrulamasından geçemedi",
                    entity_type, entity_text,
                )
                continue

            # Orijinal metinde tüm geçişleri bul
            start = 0
            while True:
                pos = text.find(entity_text, start)
                if pos == -1:
                    break
                span_key = (pos, pos + len(entity_text))
                if span_key not in seen:
                    seen.add(span_key)
                    spans.append(DetectedSpan(
                        entity_type=entity_type,
                        text=entity_text,
                        start=pos,
                        end=pos + len(entity_text),
                    ))
                start = pos + 1

        return spans

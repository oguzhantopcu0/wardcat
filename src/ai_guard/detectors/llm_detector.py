"""
LLM-based PII detector.

Performs PII detection using an on-prem Llama (or other model).
Implements the same BaseDetector interface as the regex and NER detectors,
so it is used transparently by the DetectionEngine.

Design decision: LangChain / LangGraph was NOT used.
Rationale: A direct httpx implementation for a single prompt → JSON parse →
DetectedSpan conversion is lighter, more testable, and more transparent
than 100+ transitive dependencies.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import List, Set

from ai_guard.detectors.base import BaseDetector, DetectedSpan
from ai_guard.llm.backends.base import BaseLLMBackend
from ai_guard.llm.prompt import build_messages

logger = logging.getLogger(__name__)

# For extracting the JSON array from the LLM response
_JSON_RE = re.compile(r"\[.*?\]", re.DOTALL)


@dataclass
class _CacheEntry:
    """A single TTL cache entry."""
    spans: List[DetectedSpan]
    expires_at: float

# Minimum format validation patterns for structural entities.
# If the LLM returns these types, the content must also match the format;
# otherwise the result is discarded as a hallucination.
_STRUCTURAL_VALIDATORS: dict[str, re.Pattern] = {
    # Person name must consist of at least two words (first + last name).
    # Single words (e.g. "target", "customer") are LLM hallucinations → discarded.
    "PERSON":     re.compile(r"^\S+(?:\s+\S+)+$"),
    "TC_ID":      re.compile(r"^\d{11}$"),
    "IBAN":       re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9 ]{10,}$", re.IGNORECASE),
    "CREDIT_CARD": re.compile(r"^[\d\s\-]{13,19}$"),
    "PHONE":      re.compile(r"[\d\s\-\+\(\)]{7,}"),
    "IP_ADDRESS": re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$"),
    "POSTAL_CODE": re.compile(r"^\d{5}$"),
    "UUID":        re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE),
    "SSN":         re.compile(r"^\d{3}-\d{2}-\d{4}$"),
    "MAC_ADDRESS": re.compile(r"^(?:[0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}$"),
    "JWT":         re.compile(r"^eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]*$"),
    "IPv6":        re.compile(
        r"^(?:"
        r"(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}"                   # full
        r"|(?:[0-9a-fA-F]{1,4}:){1,7}:"                                # trailing ::
        r"|:(?::[0-9a-fA-F]{1,4}){1,7}"                                # leading ::
        r"|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}"              # 1-gap
        r"|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}"    # 2-gap
        r"|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}"    # 3-gap
        r"|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}"    # 4-gap
        r"|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}"    # 5-gap
        r")$",
        re.IGNORECASE,
    ),
    "NIN":         re.compile(r"^[A-Z]{2}\d{6}[A-D]$", re.IGNORECASE),
}


class LLMDetector(BaseDetector):
    """
    PII detection via on-prem LLM.

    Supported backends:
    - OllamaBackend  — Ollama REST API (local model execution)
    - OpenAICompatBackend — vLLM, LM Studio, LocalAI, LiteLLM

    On error (connection loss, malformed JSON, etc.) logs a WARNING
    and returns an empty list — other detectors are not blocked.
    """

    def __init__(
        self,
        backend: BaseLLMBackend,
        enabled_entities: Set[str],
        *,
        timeout: int = 60,
        cache_ttl: int = 0,
    ) -> None:
        self.backend          = backend
        self.enabled_entities = enabled_entities
        self.timeout          = timeout
        self._cache_ttl       = cache_ttl   # seconds; 0 = disabled
        self._cache: dict[str, _CacheEntry] = {}
        self._cache_lock      = threading.Lock()

    def detect(self, text: str) -> List[DetectedSpan]:
        if not text.strip():
            return []

        if self._cache_ttl > 0:
            key = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()
            with self._cache_lock:
                entry = self._cache.get(key)
                if entry is not None and time.monotonic() < entry.expires_at:
                    logger.debug("LLM detector: cache hit for text of length %d", len(text))
                    return entry.spans

        messages = build_messages(text, self.enabled_entities)
        try:
            raw = self.backend.complete_messages(messages, timeout=self.timeout)
            entities = self._parse_llm_response(raw)
            spans = self._locate_spans(text, entities)
            logger.debug("LLM detector: %d entity/entities returned, %d span(s) located", len(entities), len(spans))
        except ConnectionError as exc:
            logger.warning("LLM detector connection error: %s", exc)
            return []
        except (TimeoutError, ValueError, OSError) as exc:
            logger.warning("LLM detector failed: %s", exc)
            return []

        if self._cache_ttl > 0:
            with self._cache_lock:
                self._cache[key] = _CacheEntry(
                    spans=spans,
                    expires_at=time.monotonic() + self._cache_ttl,
                )

        return spans

    async def detect_async(self, text: str) -> List[DetectedSpan]:
        """Async variant — uses the backend's native async HTTP client."""
        if not text.strip():
            return []

        if self._cache_ttl > 0:
            key = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()
            with self._cache_lock:
                entry = self._cache.get(key)
                if entry is not None and time.monotonic() < entry.expires_at:
                    logger.debug("LLM detector: cache hit for text of length %d", len(text))
                    return entry.spans
        else:
            key = None

        messages = build_messages(text, self.enabled_entities)
        try:
            raw = await self.backend.complete_messages_async(messages, timeout=self.timeout)
            entities = self._parse_llm_response(raw)
            spans = self._locate_spans(text, entities)
        except ConnectionError as exc:
            logger.warning("LLM detector connection error: %s", exc)
            return []
        except (TimeoutError, ValueError, OSError) as exc:
            logger.warning("LLM detector failed: %s", exc)
            return []

        if self._cache_ttl > 0 and key is not None:
            with self._cache_lock:
                self._cache[key] = _CacheEntry(
                    spans=spans,
                    expires_at=time.monotonic() + self._cache_ttl,
                )

        return spans

    # ------------------------------------------------------------------

    def _parse_llm_response(self, raw: str) -> list[dict]:
        """
        Extract the JSON array from the LLM response.

        Small models sometimes add ```json ... ``` blocks or
        explanatory text; these are cleaned up with regex.
        """
        # Strip markdown code block
        raw = re.sub(r"```(?:json)?", "", raw).strip()

        match = _JSON_RE.search(raw)
        if not match:
            logger.debug("No JSON array found in LLM response. Raw response: %.200r", raw)
            return []

        try:
            data = json.loads(match.group())
            if not isinstance(data, list):
                logger.debug("LLM JSON response is not a list: %r", type(data).__name__)
                return []
            return data
        except json.JSONDecodeError as exc:
            logger.debug("LLM response JSON parse error: %s — raw: %.200r", exc, raw)
            return []

    def _locate_spans(self, text: str, entities: list[dict]) -> List[DetectedSpan]:
        """
        Locate the entity texts returned by the LLM within the original text.

        The LLM sometimes slightly modifies text (case changes, etc.);
        if no match is found, that entity is skipped.
        """
        spans: List[DetectedSpan] = []
        seen: set[tuple[int, int]] = set()   # duplicate position check

        for item in entities:
            entity_type = str(item.get("type", "")).upper().strip()
            entity_text = str(item.get("text", "")).strip()

            if not entity_text or entity_type not in self.enabled_entities:
                continue

            # Format validation for structural entities:
            # Silently skip if the LLM assigned wrong content (hallucination).
            validator = _STRUCTURAL_VALIDATORS.get(entity_type)
            if validator and not validator.search(entity_text):
                logger.debug(
                    "Hallucination filter: %s %r failed format validation",
                    entity_type, entity_text,
                )
                continue

            # Find all occurrences in the original text
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
                        confidence=0.85,
                    ))
                start = pos + 1

        return spans

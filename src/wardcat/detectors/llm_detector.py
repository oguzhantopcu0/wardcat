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

import asyncio
import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass

from wardcat.detectors.base import BaseDetector, DetectedSpan
from wardcat.llm.backends.base import BaseLLMBackend
from wardcat.llm.prompt import build_messages
from wardcat.utils.text import chunk_by_paragraph

logger = logging.getLogger(__name__)

# For extracting the JSON array from the LLM response
_JSON_RE = re.compile(r"\[.*?\]", re.DOTALL)
# (paragraph chunking now lives in wardcat.utils.text.chunk_by_paragraph)


@dataclass
class _CacheEntry:
    """A single TTL cache entry."""

    spans: list[DetectedSpan]
    expires_at: float


# Minimum format validation patterns for structural entities.
# If the LLM returns these types, the content must also match the format;
# otherwise the result is discarded as a hallucination.
_STRUCTURAL_VALIDATORS: dict[str, re.Pattern] = {
    # Person name must consist of at least two words (first + last name).
    # Single words (e.g. "target", "customer") are LLM hallucinations → discarded.
    "PERSON": re.compile(r"^\S+(?:\s+\S+)+$"),
    "TC_ID": re.compile(r"^\d{11}$"),
    "IBAN": re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9 ]{10,}$", re.IGNORECASE),
    "CREDIT_CARD": re.compile(r"^[\d\s\-]{13,19}$"),
    "PHONE": re.compile(r"[\d\s\-\+\(\)]{7,}"),
    "IP_ADDRESS": re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$"),
    "POSTAL_CODE": re.compile(r"^\d{5}$"),
    "UUID": re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
    ),
    "SSN": re.compile(r"^\d{3}-\d{2}-\d{4}$"),
    "MAC_ADDRESS": re.compile(r"^(?:[0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}$"),
    "JWT": re.compile(r"^eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]*$"),
    "IPv6": re.compile(
        r"^(?:"
        r"(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}"  # full
        r"|(?:[0-9a-fA-F]{1,4}:){1,7}:"  # trailing ::
        r"|:(?::[0-9a-fA-F]{1,4}){1,7}"  # leading ::
        r"|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}"  # 1-gap
        r"|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}"  # 2-gap
        r"|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}"  # 3-gap
        r"|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}"  # 4-gap
        r"|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}"  # 5-gap
        r")$",
        re.IGNORECASE,
    ),
    "NIN": re.compile(r"^[A-Z]{2}\d{6}[A-D]$", re.IGNORECASE),
}


class LLMDetector(BaseDetector):
    """
    PII detection via on-prem LLM.

    Supported backends:
    - OllamaBackend  — Ollama REST API (local model execution)
    - OpenAICompatBackend — vLLM, LM Studio, LocalAI, LiteLLM

    On error (connection loss, malformed JSON, etc.) logs a WARNING
    and returns an empty list — other detectors are not blocked.

    Adjudication: when the engine passes ``candidates`` (spans found by the
    regex/NER detectors), the LLM both verifies those candidates and detects
    new PII in a single call. When no candidates are passed the behaviour is
    identical to pure detection, so LLM-only deployments are unaffected.
    """

    # Marks this detector as the one the engine can route candidates to.
    can_adjudicate = True

    def __init__(
        self,
        backend: BaseLLMBackend,
        enabled_entities: set[str],
        *,
        timeout: int = 60,
        cache_ttl: int = 0,
        chunk_chars: int = 800,
    ) -> None:
        self.backend = backend
        self.enabled_entities = enabled_entities
        self.timeout = timeout
        self._cache_ttl = cache_ttl  # seconds; 0 = disabled
        self._chunk_chars = chunk_chars  # max chars per LLM call; 0 = disabled
        self._cache: dict[str, _CacheEntry] = {}
        self._cache_lock = threading.Lock()

    def detect(
        self,
        text: str,
        candidates: list[DetectedSpan] | None = None,
    ) -> list[DetectedSpan]:
        if not text.strip():
            return []

        if self._cache_ttl > 0:
            key = self._cache_key(text, candidates)
            with self._cache_lock:
                entry = self._cache.get(key)
                if entry is not None and time.monotonic() < entry.expires_at:
                    logger.debug("LLM detector: cache hit for text of length %d", len(text))
                    return entry.spans
        else:
            key = None

        chunks = self._to_chunks(text)
        spans: list[DetectedSpan] = []

        for chunk_text, offset in chunks:
            if not chunk_text.strip():
                continue
            chunk_cands = self._candidates_for_chunk(candidates, offset, len(chunk_text))
            messages = build_messages(chunk_text, self.enabled_entities, chunk_cands)
            try:
                raw = self.backend.complete_messages(messages, timeout=self.timeout)
                entities = self._parse_llm_response(raw)
                chunk_spans = self._locate_spans(chunk_text, entities)
                spans.extend(self._offset_spans(chunk_spans, offset))
                logger.debug(
                    "LLM detector: chunk offset=%d len=%d → %d span(s)",
                    offset,
                    len(chunk_text),
                    len(chunk_spans),
                )
            # A ConnectionError (backend unreachable) propagates so the engine can
            # surface it on the result — the whole layer is unavailable, not just
            # this chunk. (ConnectionError is an OSError subclass, so re-raise it
            # explicitly before the transient-error catch below.)
            except ConnectionError:
                raise
            # Per-chunk transient errors are swallowed to keep scanning.
            except (TimeoutError, ValueError, OSError) as exc:
                logger.warning("LLM detector failed (offset=%d): %s", offset, exc)

        if self._cache_ttl > 0 and key is not None:
            with self._cache_lock:
                self._cache[key] = _CacheEntry(
                    spans=spans,
                    expires_at=time.monotonic() + self._cache_ttl,
                )

        return spans

    async def detect_async(
        self,
        text: str,
        candidates: list[DetectedSpan] | None = None,
    ) -> list[DetectedSpan]:
        """Async variant — chunks are scanned concurrently via asyncio.gather."""
        if not text.strip():
            return []

        if self._cache_ttl > 0:
            key = self._cache_key(text, candidates)
            with self._cache_lock:
                entry = self._cache.get(key)
                if entry is not None and time.monotonic() < entry.expires_at:
                    logger.debug("LLM detector: cache hit for text of length %d", len(text))
                    return entry.spans
        else:
            key = None

        chunks = self._to_chunks(text)

        async def _scan_chunk(chunk_text: str, offset: int) -> list[DetectedSpan]:
            if not chunk_text.strip():
                return []
            chunk_cands = self._candidates_for_chunk(candidates, offset, len(chunk_text))
            messages = build_messages(chunk_text, self.enabled_entities, chunk_cands)
            try:
                raw = await self.backend.complete_messages_async(messages, timeout=self.timeout)
                entities = self._parse_llm_response(raw)
                return self._offset_spans(self._locate_spans(chunk_text, entities), offset)
            # ConnectionError propagates (the whole layer is unavailable); the
            # engine records it as a scan warning. (It is an OSError subclass, so
            # re-raise before the transient-error catch below.)
            except ConnectionError:
                raise
            # Transient per-chunk errors are swallowed so one bad chunk does not
            # lose the rest.
            except (TimeoutError, ValueError, OSError) as exc:
                logger.warning("LLM detector failed (offset=%d): %s", offset, exc)
                return []

        results = await asyncio.gather(*[_scan_chunk(ct, off) for ct, off in chunks])
        spans: list[DetectedSpan] = [s for chunk_spans in results for s in chunk_spans]

        if self._cache_ttl > 0 and key is not None:
            with self._cache_lock:
                self._cache[key] = _CacheEntry(
                    spans=spans,
                    expires_at=time.monotonic() + self._cache_ttl,
                )

        return spans

    # ------------------------------------------------------------------

    @staticmethod
    def _candidates_for_chunk(
        candidates: list[DetectedSpan] | None,
        offset: int,
        chunk_len: int,
    ) -> list[tuple[str, str]] | None:
        """Select candidates that fall within a chunk → ``(type, text)`` pairs.

        Positions are not needed by the prompt, only the type and surface text,
        so chunk-local offsets are irrelevant here — we just filter by range.
        """
        if not candidates:
            return None
        end = offset + chunk_len
        return [(s.entity_type, s.text) for s in candidates if s.start >= offset and s.end <= end]

    def _cache_key(self, text: str, candidates: list[DetectedSpan] | None) -> str:
        """Cache key over text + candidate set (different candidates → different verdict)."""
        # MD5 is fine here: a non-cryptographic cache key, not a security hash.
        h = hashlib.md5(text.encode("utf-8", errors="replace"), usedforsecurity=False)
        if candidates:
            for s in sorted(candidates, key=lambda c: (c.start, c.end, c.entity_type)):
                h.update(f"|{s.entity_type}:{s.start}:{s.end}".encode("utf-8", errors="replace"))
        return h.hexdigest()

    def _to_chunks(self, text: str) -> list[tuple[str, int]]:
        """Split text into (chunk_text, start_offset) pairs (see chunk_by_paragraph).

        Small LLMs lose focus on entity names when given very long texts — chunking
        keeps each LLM call within an attention-friendly size.
        """
        return chunk_by_paragraph(text, self._chunk_chars)

    @staticmethod
    def _offset_spans(spans: list[DetectedSpan], offset: int) -> list[DetectedSpan]:
        """Shift span positions by offset to map chunk positions back to full text."""
        if offset == 0:
            return spans
        return [
            DetectedSpan(
                entity_type=s.entity_type,
                text=s.text,
                start=s.start + offset,
                end=s.end + offset,
                confidence=s.confidence,
            )
            for s in spans
        ]

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

    def _locate_spans(self, text: str, entities: list[dict]) -> list[DetectedSpan]:
        """
        Locate the entity texts returned by the LLM within the original text.

        The LLM sometimes slightly modifies text (case changes, etc.);
        if no match is found, that entity is skipped.
        """
        spans: list[DetectedSpan] = []
        seen: set[tuple[int, int]] = set()  # duplicate position check

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
                    entity_type,
                    entity_text,
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
                    spans.append(
                        DetectedSpan(
                            entity_type=entity_type,
                            text=entity_text,
                            start=pos,
                            end=pos + len(entity_text),
                            confidence=0.85,
                        )
                    )
                start = pos + 1

        return spans

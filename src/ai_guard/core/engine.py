from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import TYPE_CHECKING, Any, cast

from ai_guard.core.models import Action, ScanResult, Violation
from ai_guard.detectors.base import BaseDetector, DetectedSpan
from ai_guard.utils.hashing import sha256_hash

if TYPE_CHECKING:
    from ai_guard.detectors.llm_detector import LLMDetector

logger = logging.getLogger(__name__)

# Default upper limit for safe input size.
# If exceeded, scan() raises ValueError — does not lock the regex engine.
_MAX_TEXT_BYTES = 500_000


def _mask_value(entity_type: str, text: str) -> str:
    """Produce an entity-aware masked version of *text*.

    Masking rules per entity type:

    ============== =================================================
    CREDIT_CARD    Last 4 digits visible: ``************1111``
    EMAIL          First char + stars + full domain: ``u***@example.com``
    PHONE          Last 4 digits visible: ``*******5678``
    SSN            Standard US format: ``***-**-6789``
    IBAN           Country code + last 4: ``TR**...**1326``
    TC_ID          Last 3 digits: ``********950``
    NIN            Last 3: ``AB123***``
    *default*      First 2 + stars + last 2: ``ab****cd``
    ============== =================================================
    """
    n = len(text)
    if entity_type == "CREDIT_CARD":
        digits = re.sub(r"[^0-9]", "", text)
        if len(digits) >= 4:
            return "*" * (len(digits) - 4) + digits[-4:]
        return "*" * n

    if entity_type == "EMAIL":
        at = text.find("@")
        if at > 0:
            local = text[:at]
            domain = text[at:]
            masked_local = local[0] + "*" * max(len(local) - 1, 1)
            return masked_local + domain
        return "*" * n

    if entity_type == "PHONE":
        digits = re.sub(r"[^0-9]", "", text)
        if len(digits) >= 4:
            return "*" * (n - 4) + text[-4:]
        return "*" * n

    if entity_type == "SSN":
        digits = re.sub(r"[^0-9]", "", text)
        if len(digits) >= 4:
            return f"***-**-{digits[-4:]}"
        return "*" * n

    if entity_type == "IBAN":
        # Keep country code (2 chars) + last 4
        if n >= 6:
            return text[:2] + "*" * (n - 6) + text[-4:]
        return "*" * n

    if entity_type == "TC_ID":
        # Last 3 digits visible
        if n >= 3:
            return "*" * (n - 3) + text[-3:]
        return "*" * n

    if entity_type == "NIN":
        # Last 3 chars visible
        if n >= 3:
            return "*" * (n - 3) + text[-3:]
        return "*" * n

    # Default: first 2 + stars + last 2
    if n >= 4:
        return text[:2] + "*" * (n - 4) + text[-2:]
    return "*" * n


class DetectionEngine:
    """
    Merges spans from all detectors, resolves overlaps,
    applies configured actions, and returns a ScanResult.
    """

    def __init__(self, config: dict[str, Any], detectors: list[BaseDetector]) -> None:
        self.config = config
        self.detectors = detectors
        # Ensemble adjudication: when enabled and an LLM detector is present,
        # regex/NER spans are passed to the LLM as candidates to verify/relabel/
        # drop (one combined detection + adjudication call). Deterministic
        # (confidence >= 1.0) regex spans are always kept regardless of the LLM.
        self._llm_detectors: list[LLMDetector] = cast(
            "list[LLMDetector]",
            [d for d in detectors if getattr(d, "can_adjudicate", False)],
        )
        self._other_detectors = [d for d in detectors if not getattr(d, "can_adjudicate", False)]
        self._use_adjudication = bool(
            config.get("llm_detector", {}).get("adjudicate", False)
        ) and bool(self._llm_detectors)
        self.salt: str = config.get("salt", "")
        self.entity_config: dict[str, Any] = config.get("entities", {})
        self._max_text_bytes: int = config.get("max_text_bytes", _MAX_TEXT_BYTES)
        self._allowlist: set[str] = set(config.get("allowlist", []))
        self._denylist: list[dict[str, str]] = config.get("denylist", [])

        if not self.salt:
            logger.debug(
                "Hash salt is empty — identical values will produce the same hash. "
                "Set the LLMGUARD_SALT environment variable in production."
            )

    # ------------------------------------------------------------------
    # Public scan API
    # ------------------------------------------------------------------

    def scan(self, text: str) -> ScanResult:
        """Run all detectors, apply actions, and return the result."""
        t_start = time.perf_counter()
        self._check_size(text)

        if self._use_adjudication:
            candidate_spans: list[DetectedSpan] = []
            for detector in self._other_detectors:
                candidate_spans.extend(detector.detect(text))
            raw_spans = [s for s in candidate_spans if s.confidence >= 1.0]
            for llm in self._llm_detectors:
                raw_spans.extend(llm.detect(text, candidates=candidate_spans))
        else:
            raw_spans = []
            for detector in self.detectors:
                t_det = time.perf_counter()
                spans = detector.detect(text)
                logger.debug(
                    "%s: %d span(s) detected (%.1f ms)",
                    type(detector).__name__,
                    len(spans),
                    (time.perf_counter() - t_det) * 1000,
                )
                raw_spans.extend(spans)

        raw_spans.extend(self._collect_denylist_spans(text))
        spans = self._filter_spans(raw_spans)
        sanitized, violations = self._apply_actions(text, spans)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "scan completed: %d violation(s), %d character(s), %.1f ms",
            len(violations),
            len(text),
            elapsed_ms,
        )
        return ScanResult(original_text=text, sanitized_text=sanitized, violations=violations)

    async def scan_async(self, text: str) -> ScanResult:
        """Async variant — uses native async for I/O-bound detectors (LLM backend).

        CPU-bound detectors (regex, NER) run via ``asyncio.to_thread``;
        the LLM detector uses its own ``detect_async()`` method with a
        native ``httpx.AsyncClient``.
        """
        t_start = time.perf_counter()
        self._check_size(text)

        async def _run(detector: BaseDetector) -> list[DetectedSpan]:
            if hasattr(detector, "detect_async"):
                return await detector.detect_async(text)
            return await asyncio.to_thread(detector.detect, text)

        if self._use_adjudication:
            cand_results = await asyncio.gather(*(_run(d) for d in self._other_detectors))
            candidate_spans: list[DetectedSpan] = [s for batch in cand_results for s in batch]
            raw_spans = [s for s in candidate_spans if s.confidence >= 1.0]
            llm_results = await asyncio.gather(
                *(llm.detect_async(text, candidates=candidate_spans) for llm in self._llm_detectors)
            )
            for batch in llm_results:
                raw_spans.extend(batch)
        else:
            results = await asyncio.gather(*(_run(d) for d in self.detectors))
            raw_spans = [s for batch in results for s in batch]
        raw_spans.extend(self._collect_denylist_spans(text))
        spans = self._filter_spans(raw_spans)
        sanitized, violations = self._apply_actions(text, spans)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "scan_async completed: %d violation(s), %d character(s), %.1f ms",
            len(violations),
            len(text),
            elapsed_ms,
        )
        return ScanResult(original_text=text, sanitized_text=sanitized, violations=violations)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_size(self, text: str) -> None:
        byte_len = len(text.encode("utf-8", errors="replace"))
        if byte_len > self._max_text_bytes:
            raise ValueError(
                f"Input text is too large: {byte_len:,} bytes "
                f"(maximum: {self._max_text_bytes:,} bytes). "
                "Split the text into smaller chunks."
            )

    def _filter_spans(self, raw_spans: list[DetectedSpan]) -> list[DetectedSpan]:
        """Sort, resolve overlaps, and apply allowlist filter."""
        spans = self._resolve_overlaps(sorted(raw_spans, key=lambda s: s.start))
        if self._allowlist:
            spans = [s for s in spans if s.text not in self._allowlist]
        return spans

    def _compute_replacement(self, action: Action, span: DetectedSpan) -> str | None:
        """Return the replacement string for *span* under *action*, or None for WARN."""
        if action == Action.HASH:
            digest = sha256_hash(span.text, self.salt)[:16]
            return f"[{span.entity_type}:{digest}]"
        if action == Action.REDACT:
            return f"[{span.entity_type}]"
        if action == Action.MASK:
            return _mask_value(span.entity_type, span.text)
        return None  # WARN

    def _apply_actions(self, text: str, spans: list[DetectedSpan]) -> tuple[str, list[Violation]]:
        """Apply configured actions to *spans* and return (sanitized_text, violations)."""
        violations: list[Violation] = []
        sanitized = text
        offset = 0

        for span in spans:
            entity_cfg = self.entity_config.get(span.entity_type, {})
            action = Action(entity_cfg.get("action", "warn"))
            replacement = self._compute_replacement(action, span)

            if replacement is not None:
                adj_start = span.start + offset
                adj_end = span.end + offset
                sanitized = sanitized[:adj_start] + replacement + sanitized[adj_end:]
                offset += len(replacement) - (span.end - span.start)

            violations.append(
                Violation(
                    entity_type=span.entity_type,
                    original=span.text,
                    start=span.start,
                    end=span.end,
                    action=action,
                    replacement=replacement,
                    confidence=span.confidence,
                )
            )

        return sanitized, violations

    def _collect_denylist_spans(self, text: str) -> list[DetectedSpan]:
        """Match denylist entries (exact value or regex pattern) against *text*."""
        spans: list[DetectedSpan] = []
        for entry in self._denylist:
            entity_type = entry.get("entity_type", "CUSTOM")

            if "pattern" in entry:
                # Regex denylist entry
                try:
                    compiled = re.compile(entry["pattern"])
                except re.error:
                    logger.warning("Denylist pattern %r is invalid — skipped.", entry["pattern"])
                    continue
                for m in compiled.finditer(text):
                    spans.append(
                        DetectedSpan(
                            entity_type=entity_type,
                            text=m.group(),
                            start=m.start(),
                            end=m.end(),
                            confidence=1.0,
                        )
                    )

            elif "value" in entry:
                # Exact-match denylist entry
                value = entry["value"]
                if not value:
                    continue
                start = 0
                while True:
                    pos = text.find(value, start)
                    if pos == -1:
                        break
                    spans.append(
                        DetectedSpan(
                            entity_type=entity_type,
                            text=value,
                            start=pos,
                            end=pos + len(value),
                            confidence=1.0,
                        )
                    )
                    start = pos + 1

        return spans

    def _resolve_overlaps(self, spans: list[DetectedSpan]) -> list[DetectedSpan]:
        """Keeps the longer span when overlapping spans are found."""
        if not spans:
            return spans
        result: list[DetectedSpan] = [spans[0]]
        for span in spans[1:]:
            last = result[-1]
            if span.start < last.end:
                if (span.end - span.start) > (last.end - last.start):
                    result[-1] = span
            else:
                result.append(span)
        return result

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from ai_guard.core.models import Action, ScanResult, Violation
from ai_guard.detectors.base import BaseDetector, DetectedSpan
from ai_guard.utils.hashing import sha256_hash

logger = logging.getLogger(__name__)

# Upper limit for safe input size.
# If exceeded, scan() raises ValueError — does not lock the regex engine.
_MAX_TEXT_BYTES = 500_000


class DetectionEngine:
    """
    Merges spans from all detectors, resolves overlaps,
    applies configured actions, and returns a ScanResult.
    """

    def __init__(self, config: Dict[str, Any], detectors: List[BaseDetector]) -> None:
        self.config = config
        self.detectors = detectors
        self.salt: str = config.get("salt", "")
        self.entity_config: Dict[str, Any] = config.get("entities", {})

        if not self.salt:
            logger.debug(
                "Hash salt is empty — identical values will produce the same hash. "
                "Set the LLMGUARD_SALT environment variable in production."
            )

    def scan(self, text: str) -> ScanResult:
        """Run all detectors, apply actions, and return the result."""
        t_start = time.perf_counter()

        # Hard input size limit — reject if exceeded (DoS protection)
        byte_len = len(text.encode("utf-8", errors="replace"))
        if byte_len > _MAX_TEXT_BYTES:
            raise ValueError(
                f"Input text is too large: {byte_len:,} bytes "
                f"(maximum: {_MAX_TEXT_BYTES:,} bytes). "
                "Split the text into smaller chunks."
            )

        # 1. Collect spans from all detectors
        raw_spans: List[DetectedSpan] = []
        for detector in self.detectors:
            detector_name = type(detector).__name__
            t_det = time.perf_counter()
            spans = detector.detect(text)
            logger.debug(
                "%s: %d span(s) detected (%.1f ms)",
                detector_name,
                len(spans),
                (time.perf_counter() - t_det) * 1000,
            )
            raw_spans.extend(spans)

        # 2. Sort by position and resolve overlaps
        spans = self._resolve_overlaps(sorted(raw_spans, key=lambda s: s.start))

        # 3. Apply actions
        violations: List[Violation] = []
        sanitized = text
        offset = 0  # track position shift as text is modified

        for span in spans:
            entity_cfg = self.entity_config.get(span.entity_type, {})
            action = Action(entity_cfg.get("action", "warn"))

            replacement: str | None = None
            if action == Action.HASH:
                digest = sha256_hash(span.text, self.salt)[:16]
                replacement = f"[{span.entity_type}:{digest}]"
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
                )
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "scan completed: %d violation(s), %d character(s), %.1f ms",
            len(violations),
            len(text),
            elapsed_ms,
        )

        return ScanResult(
            original_text=text,
            sanitized_text=sanitized,
            violations=violations,
        )

    # ------------------------------------------------------------------

    def _resolve_overlaps(self, spans: List[DetectedSpan]) -> List[DetectedSpan]:
        """Keeps the longer span when overlapping spans are found."""
        if not spans:
            return spans
        result: List[DetectedSpan] = [spans[0]]
        for span in spans[1:]:
            last = result[-1]
            if span.start < last.end:          # overlap detected
                if (span.end - span.start) > (last.end - last.start):
                    result[-1] = span          # keep the longer one
            else:
                result.append(span)
        return result

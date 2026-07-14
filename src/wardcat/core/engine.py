from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from wardcat.core.anonymizer import Anonymizer
from wardcat.core.models import ScanResult
from wardcat.detectors.base import BaseDetector, DetectedSpan

logger = logging.getLogger(__name__)

# Default upper limit for safe input size.
# If exceeded, scan() raises ValueError — does not lock the regex engine.
_MAX_TEXT_BYTES = 500_000

# In adjudication, spans at or above this confidence are always kept regardless
# of the LLM verdict. It is set to the lowest regex tier (fuzzy = 0.90) so that
# ALL regex layers — including a fuzzy ADDRESS match — are protected, while the
# model layers (NER/LLM 0.85) are candidates the LLM may
# drop/relabel. For a PII tool, never letting the LLM drop a deterministic match
# is the safe default: over-redaction beats a leak.
_ADJUDICATION_KEEP_CONFIDENCE = 0.90


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
        # drop (one combined detection + adjudication call). Every regex span
        # (confidence >= _ADJUDICATION_KEEP_CONFIDENCE) is always kept regardless
        # of the LLM — a weak adjudicator that fails to re-detect a real match
        # would otherwise leak it. Only the model layers (NER/LLM) are
        # candidates the LLM may drop/relabel.
        # Detectors are addressed only through BaseDetector — the engine never
        # imports a concrete detector. Adjudicators advertise themselves via the
        # can_adjudicate flag.
        self._adjudicators = [d for d in detectors if d.can_adjudicate]
        self._other_detectors = [d for d in detectors if not d.can_adjudicate]
        self._use_adjudication = bool(
            config.get("llm_detector", {}).get("adjudicate", False)
        ) and bool(self._adjudicators)
        self.salt: str = config.get("salt", "")
        self.entity_config: dict[str, Any] = config.get("entities", {})
        self._max_text_bytes: int = config.get("max_text_bytes", _MAX_TEXT_BYTES)
        self._allowlist: set[str] = set(config.get("allowlist", []))
        self._denylist: list[dict[str, str]] = config.get("denylist", [])
        # Value propagation: once any layer detects a value, redact every other
        # whole-token occurrence of that exact value too. Closes the gap where a
        # model-based layer (NER/LLM) reports a repeated value only once.
        # Opt-in — it can over-redact, so short values are skipped and matches
        # must be token-bounded.
        self._propagate: bool = bool(config.get("propagate_matches", False))
        self._propagate_min_len: int = config.get("propagate_min_length", 3)
        # Detection (this class) is kept separate from anonymization (applying the
        # configured action to each span); the Anonymizer owns that stage.
        self._anonymizer = Anonymizer(self.entity_config, self.salt)

        if not self.salt:
            logger.debug(
                "Hash salt is empty — identical values will produce the same hash. "
                "Pass salt=... to Wardcat(...) in production."
            )

    # ------------------------------------------------------------------
    # Public scan API
    # ------------------------------------------------------------------

    def scan(self, text: str) -> ScanResult:
        """Run all detectors, apply actions, and return the result."""
        t_start = time.perf_counter()
        self._check_size(text)

        warnings: list[str] = []
        if self._use_adjudication:
            candidate_spans: list[DetectedSpan] = []
            for detector in self._other_detectors:
                candidate_spans.extend(self._safe_detect(detector, text, warnings))
            raw_spans = [
                s for s in candidate_spans if s.confidence >= _ADJUDICATION_KEEP_CONFIDENCE
            ]
            for adjudicator in self._adjudicators:
                raw_spans.extend(
                    self._safe_detect(adjudicator, text, warnings, candidates=candidate_spans)
                )
        else:
            raw_spans = []
            for detector in self.detectors:
                raw_spans.extend(self._safe_detect(detector, text, warnings))

        raw_spans.extend(self._collect_denylist_spans(text))
        spans = self._filter_spans(raw_spans, text)
        sanitized, violations = self._anonymizer.apply(text, spans)

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
            warnings=warnings,
            _salt=self.salt,
        )

    async def scan_async(self, text: str) -> ScanResult:
        """Async variant — uses native async for I/O-bound detectors (LLM backend).

        CPU-bound detectors (regex, NER) run via ``asyncio.to_thread``;
        the LLM detector uses its own ``detect_async()`` method with a
        native ``httpx.AsyncClient``.
        """
        t_start = time.perf_counter()
        self._check_size(text)

        warnings: list[str] = []
        if self._use_adjudication:
            cand_results = await asyncio.gather(
                *(self._safe_detect_async(d, text) for d in self._other_detectors)
            )
            candidate_spans: list[DetectedSpan] = [s for spans, _ in cand_results for s in spans]
            warnings.extend(w for _, w in cand_results if w)
            raw_spans = [
                s for s in candidate_spans if s.confidence >= _ADJUDICATION_KEEP_CONFIDENCE
            ]
            adj_results = await asyncio.gather(
                *(
                    self._safe_detect_async(d, text, candidates=candidate_spans)
                    for d in self._adjudicators
                )
            )
            for spans, w in adj_results:
                raw_spans.extend(spans)
                if w:
                    warnings.append(w)
        else:
            results = await asyncio.gather(
                *(self._safe_detect_async(d, text) for d in self.detectors)
            )
            raw_spans = [s for spans, _ in results for s in spans]
            warnings.extend(w for _, w in results if w)
        raw_spans.extend(self._collect_denylist_spans(text))
        spans = self._filter_spans(raw_spans, text)
        sanitized, violations = self._anonymizer.apply(text, spans)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "scan_async completed: %d violation(s), %d character(s), %.1f ms",
            len(violations),
            len(text),
            elapsed_ms,
        )
        return ScanResult(
            original_text=text,
            sanitized_text=sanitized,
            violations=violations,
            warnings=warnings,
            _salt=self.salt,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_detect(
        self,
        detector: BaseDetector,
        text: str,
        warnings: list[str],
        candidates: list[DetectedSpan] | None = None,
    ) -> list[DetectedSpan]:
        """Run one detector, turning a failure into a warning instead of aborting.

        A layer that cannot run (e.g. the LLM backend is unreachable) must not
        block the others — its error is recorded on the result so the caller
        knows detection was degraded rather than silently missing PII.
        """
        try:
            if candidates is None:
                return detector.detect(text)
            return detector.detect(text, candidates=candidates)
        except Exception as exc:
            msg = f"{type(detector).__name__} did not run: {exc}"
            logger.warning("scan: detector layer skipped — %s", msg)
            warnings.append(msg)
            return []

    async def _safe_detect_async(
        self,
        detector: BaseDetector,
        text: str,
        candidates: list[DetectedSpan] | None = None,
    ) -> tuple[list[DetectedSpan], str | None]:
        """Async :meth:`_safe_detect` — returns ``(spans, warning_or_None)``."""
        try:
            if candidates is None:
                return await detector.detect_async(text), None
            return await detector.detect_async(text, candidates=candidates), None
        except Exception as exc:
            msg = f"{type(detector).__name__} did not run: {exc}"
            logger.warning("scan_async: detector layer skipped — %s", msg)
            return [], msg

    def _check_size(self, text: str) -> None:
        byte_len = len(text.encode("utf-8", errors="replace"))
        if byte_len > self._max_text_bytes:
            raise ValueError(
                f"Input text is too large: {byte_len:,} bytes "
                f"(maximum: {self._max_text_bytes:,} bytes). "
                "Split the text into smaller chunks."
            )

    def _filter_spans(self, raw_spans: list[DetectedSpan], text: str) -> list[DetectedSpan]:
        """Resolve overlaps (returns start-sorted spans), apply the allowlist, and
        optionally propagate each detected value to its other occurrences."""
        spans = self._resolve_overlaps(raw_spans)
        if self._allowlist:
            spans = [s for s in spans if s.text not in self._allowlist]
        if self._propagate:
            spans = self._propagate_values(spans, text)
        return spans

    def _propagate_values(self, spans: list[DetectedSpan], text: str) -> list[DetectedSpan]:
        """Add a span for every other whole-token occurrence of each detected value.

        A model-based layer often reports a repeated value only once; this fills
        in the misses so every occurrence is anonymized. Only exact, token-bounded
        matches of values at least ``propagate_min_length`` chars long are added,
        to avoid over-redacting short or substring matches. New spans inherit the
        original's entity type/action; overlaps are re-resolved so a propagated
        match never displaces a stronger (e.g. checksum-regex) span.
        """
        if not spans:
            return spans
        occupied = {(s.start, s.end) for s in spans}
        # One value → the highest-confidence span for it (so propagated copies
        # inherit the best entity type/confidence when a value was seen twice).
        best: dict[str, DetectedSpan] = {}
        for s in spans:
            if len(s.text) < self._propagate_min_len:
                continue
            cur = best.get(s.text)
            if cur is None or s.confidence > cur.confidence:
                best[s.text] = s

        extra: list[DetectedSpan] = []
        for value, template in best.items():
            for m in re.finditer(re.escape(value), text):
                start, end = m.start(), m.end()
                if (start, end) in occupied or not self._token_bounded(text, start, end):
                    continue
                occupied.add((start, end))
                extra.append(
                    DetectedSpan(
                        entity_type=template.entity_type,
                        text=value,
                        start=start,
                        end=end,
                        confidence=template.confidence,
                    )
                )
        if not extra:
            return spans
        return self._resolve_overlaps(spans + extra)

    @staticmethod
    def _token_bounded(text: str, start: int, end: int) -> bool:
        """True if the match is not glued to an alphanumeric character on either side."""
        before = text[start - 1] if start > 0 else " "
        after = text[end] if end < len(text) else " "
        return not before.isalnum() and not after.isalnum()

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
        """Select a non-overlapping subset, preferring the strongest span.

        The anonymizer requires non-overlapping spans, so when candidates
        overlap we must drop all but one. Spans are ranked and accepted greedily:
        a candidate is kept only if it overlaps none of the already-accepted
        (higher-ranked) spans. Ranking, most-preferred first:

        1. **Confidence** — a checksum/regex span (``1.0``) beats a fuzzy NER/LLM
           span (``0.85``) even when the latter is longer, so a Luhn-validated
           card is never lost to an overlapping ADDRESS guess.
        2. **Length** — the longer span wins among equal confidence.
        3. **Start** — earlier first, purely for a stable, deterministic result.

        Every candidate is checked against *all* kept spans (not just the last),
        so chained/nested overlaps cannot slip a span through.
        """
        if not spans:
            return spans
        ranked = sorted(
            spans,
            key=lambda s: (-s.confidence, -(s.end - s.start), s.start),
        )
        kept: list[DetectedSpan] = []
        for span in ranked:
            if any(span.start < k.end and k.start < span.end for k in kept):
                continue  # overlaps an already-accepted, higher-ranked span
            kept.append(span)
        kept.sort(key=lambda s: s.start)
        return kept

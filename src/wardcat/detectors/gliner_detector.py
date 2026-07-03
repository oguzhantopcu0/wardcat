"""GLiNER2-based zero-shot NER detector.

GLiNER (Generalist and Lightweight model for NER) is a bidirectional-encoder
model that extracts arbitrary, prompt-given entity types — a middle ground
between SpaCy NER (fast, fixed labels) and an on-prem LLM (slow, contextual).
This detector wraps the PII-tuned GLiNER2 model and maps its labels onto
wardcat entity types via :data:`~wardcat.core.registry.GLINER_LABEL_MAP`.

The ``gliner2`` package (and its ``[local]`` extra, which pulls in torch) is an
optional dependency; it is imported lazily so the base install stays small.
Install it with ``pip install "wardcat[gliner]"``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from wardcat.core.registry import GLINER_LABEL_MAP
from wardcat.detectors.base import BaseDetector, DetectedSpan

logger = logging.getLogger(__name__)

# The PII-tuned multilingual GLiNER2 model (Apache-2.0). Covers EN/FR/ES/DE/IT/
# PT/NL — not Turkish; keep the regex/LLM layers for Turkish text.
DEFAULT_GLINER_MODEL = "fastino/gliner2-privacy-filter-PII-multi"

# GLiNER is model-based (not deterministic), so its spans are capped below every
# regex tier (checksum 1.0, structural 0.97, fuzzy 0.90) — a regex match always
# wins an overlap and is never dropped by adjudication — while staying above
# SpaCy NER / the LLM (0.85) so the richer model wins when only those overlap.
_MAX_CONFIDENCE = 0.88

# GLiNER has a fixed maximum input length (~512 tokens); text beyond it is
# silently truncated by the model. To scan long documents we split the input
# into overlapping character windows and re-base each chunk's spans. The window
# is deliberately conservative (chars, not tokens) so it holds for any language.
DEFAULT_CHUNK_SIZE = 1500
_CHUNK_OVERLAP = 120  # carry-over so an entity on a boundary is still seen whole


# ── Model singleton cache ─────────────────────────────────────────────────────
# A GLiNER2 model is loaded once per (name, quantize) key and shared across
# Wardcat instances. Thread-safe via _CACHE_LOCK.
_MODEL_CACHE: dict[tuple[str, bool], Any] = {}
_CACHE_LOCK = threading.Lock()
_COMPAT_PATCHED = False


def _ensure_transformers_compat() -> None:
    """Tolerate a list-valued ``extra_special_tokens`` in the tokenizer config.

    Some GLiNER2 model repos ship ``extra_special_tokens`` as a *list* of
    structural tokens, but transformers >= 4.44 expects a ``{name: token}``
    dict and calls ``.keys()`` on it — raising ``AttributeError`` on load. We
    patch the one method to accept a list (treating each token as its own key),
    once, and only when GLiNER is actually loaded. Applied lazily so users who
    never touch GLiNER are unaffected.
    """
    global _COMPAT_PATCHED
    if _COMPAT_PATCHED:
        return
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _B

        _orig = _B._set_model_specific_special_tokens

        def _patched(self: Any, special_tokens: Any) -> Any:
            if isinstance(special_tokens, list):
                special_tokens = {t: t for t in special_tokens}
                self.extra_special_tokens = special_tokens
            return _orig(self, special_tokens)

        # setattr (not direct assignment) so mypy doesn't flag a method override
        # when transformers is installed — the shim is deliberate.
        setattr(_B, "_set_model_specific_special_tokens", _patched)  # noqa: B010
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not apply transformers extra_special_tokens shim: %s", exc)
    _COMPAT_PATCHED = True


def _load_gliner_model(model_name: str, *, quantize: bool = False) -> Any:
    """Return the GLiNER2 model from cache; load and cache it if not present."""
    key = (model_name, quantize)
    with _CACHE_LOCK:
        if key not in _MODEL_CACHE:
            _ensure_transformers_compat()
            from gliner2 import GLiNER2  # lazy import — gliner2 is optional

            logger.info("Loading GLiNER2 model: %s (quantize=%s)", model_name, quantize)
            _MODEL_CACHE[key] = GLiNER2.from_pretrained(model_name, quantize=quantize)
            logger.info("GLiNER2 model ready: %s", model_name)
        return _MODEL_CACHE[key]


def _chunks(text: str, size: int, overlap: int):
    """Yield ``(offset, chunk_text)`` windows, breaking at whitespace when possible."""
    if len(text) <= size:
        yield 0, text
        return
    # Keep overlap well under the window so each step makes real progress.
    overlap = min(overlap, size // 2)
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Back off to the last whitespace so an entity isn't split mid-token.
            ws = text.rfind(" ", start, end)
            if ws > start:
                end = ws
        yield start, text[start:end]
        if end >= n:
            break
        start = max(end - overlap, start + 1)


class GLiNERDetector(BaseDetector):
    """Zero-shot NER detector backed by a GLiNER2 model.

    Only the GLiNER labels whose mapped entity type is enabled are requested
    from the model, and each returned span is filtered by ``threshold`` before
    being emitted. Long inputs are split into overlapping windows so the model's
    fixed maximum length does not silently truncate the document — duplicate
    spans from overlapping windows are collapsed later by the engine's overlap
    resolver.
    """

    def __init__(
        self,
        enabled_entities: set[str],
        *,
        model: str = DEFAULT_GLINER_MODEL,
        threshold: float = 0.5,
        quantize: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        self.enabled_entities = set(enabled_entities)
        self.threshold = threshold
        self.chunk_size = chunk_size
        # Ask the model only for labels that map onto an enabled entity type.
        self._labels = [
            label for label, entity in GLINER_LABEL_MAP.items() if entity in self.enabled_entities
        ]
        self.model = _load_gliner_model(model, quantize=quantize)

    def detect(self, text: str, candidates: list[DetectedSpan] | None = None) -> list[DetectedSpan]:
        """Return spans for the enabled entity types found by the GLiNER2 model."""
        if not self._labels:
            return []

        spans: list[DetectedSpan] = []
        for offset, chunk in _chunks(text, self.chunk_size, _CHUNK_OVERLAP):
            spans.extend(self._extract(chunk, offset))
        return spans

    def _extract(self, text: str, offset: int) -> list[DetectedSpan]:
        """Run the model on one window and re-base its spans by ``offset``."""
        result = self.model.extract_entities(
            text,
            self._labels,
            include_confidence=True,
            include_spans=True,
        )

        spans: list[DetectedSpan] = []
        # Return shape: {"entities": {label: [{"text","confidence","start","end"}, ...]}}
        for label, items in (result or {}).get("entities", {}).items():
            mapped = GLINER_LABEL_MAP.get(label)
            if not mapped or mapped not in self.enabled_entities:
                continue
            for item in items:
                confidence = item.get("confidence", 1.0)
                if confidence < self.threshold:
                    continue
                start, end = item.get("start"), item.get("end")
                if start is None or end is None:
                    # Without character offsets we cannot anonymize the span.
                    continue
                spans.append(
                    DetectedSpan(
                        entity_type=mapped,
                        text=item.get("text", text[start:end]),
                        start=start + offset,
                        end=end + offset,
                        confidence=min(confidence, _MAX_CONFIDENCE),
                    )
                )
        return spans

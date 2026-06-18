from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List

logger = logging.getLogger(__name__)


class Action(str, Enum):
    """Action to apply to detected PII."""

    WARN = "warn"
    """Leave the text as-is, report only as a violation."""
    HASH = "hash"
    """Mask with SHA-256 + salt: ``[ENTITY_TYPE:abcd1234]``."""
    REDACT = "redact"
    """Replace with a plain label: ``[ENTITY_TYPE]`` — no hash, no original value."""
    MASK = "mask"
    """Partially obscure: show first 2 and last 2 characters, replace the rest with ``*``.
    E.g. ``4111111111111111`` → ``41************11``.  For values shorter than 4 characters
    the entire value is replaced with ``*`` characters."""


# Known entity types — for typo checking and IDE support.
# A warning is issued if a type not in this list is configured.
KNOWN_ENTITY_TYPES: frozenset[str] = frozenset({
    "PERSON", "ORG", "EMAIL", "PHONE", "CREDIT_CARD", "IBAN",
    "TC_ID", "IP_ADDRESS", "IPv6", "ADDRESS", "POSTAL_CODE", "CUSTOM_SECRET",
    "UUID", "SSN", "MAC_ADDRESS", "JWT", "NIN",
    "UK_POSTAL_CODE", "US_ZIP_CODE", "EU_NATIONAL_ID", "PASSPORT",
    "CODICE_FISCALE", "DATE_OF_BIRTH", "VEHICLE_PLATE", "FINANCIAL_AMOUNT",
    "VAT_NUMBER",
})


def warn_unknown_entity(entity_type: str) -> None:
    """Warn when an unknown entity type is used."""
    if entity_type not in KNOWN_ENTITY_TYPES:
        logger.warning(
            "Unknown entity type: %r — this type is not recognized. "
            "Typo? Valid types: %s",
            entity_type,
            sorted(KNOWN_ENTITY_TYPES),
        )


@dataclass
class Violation:
    """A single PII violation detected in the text."""

    entity_type: str
    """E.g. ``"EMAIL"``, ``"CREDIT_CARD"``, ``"PERSON"``."""
    original: str
    """Raw value from the original text."""
    start: int
    """Start index in the original text."""
    end: int
    """End index in the original text."""
    action: Action
    """Action that was applied."""
    replacement: str | None = None
    """Placeholder produced by the hash action; ``None`` for warn."""
    confidence: float = 1.0
    """Detection confidence in [0.0, 1.0].

    - Regex-based (structural patterns): ``1.0`` — deterministic, no false positives.
    - Checksum-validated (IBAN, TC_ID): ``1.0`` — mathematically verified.
    - NER (SpaCy): ``0.85`` — model-based, occasionally mis-labels entities.
    - LLM: ``0.85`` — passed hallucination filter, but still model-based.

    Use this field to apply confidence thresholds in post-processing::

        high_confidence = [v for v in result.violations if v.confidence >= 1.0]
    """


@dataclass
class ScanResult:
    """Result of a single ``guard.scan()`` call.

    .. warning::
        The ``original_text`` and ``violations[].original`` fields contain raw PII.
        When writing this object to logs, databases, or API responses, use only
        ``sanitized_text``. Use the :meth:`redacted` method to obtain a dict
        that contains no PII.
    """

    original_text: str
    """Unmodified original input. **Contains raw PII — do not expose externally.**"""
    sanitized_text: str
    """Output text with PII masked/reported."""
    violations: List[Violation] = field(default_factory=list)
    """List of all detected violations. The ``original`` fields contain raw PII."""
    scan_error: str | None = None
    """Set when this item failed during :meth:`~ai_guard.LLMGuard.scan_batch`.
    The original text is returned unchanged. Non-None means the scan result
    is incomplete — callers should not treat the text as clean."""

    @property
    def is_clean(self) -> bool:
        """``True`` if no PII was detected."""
        return len(self.violations) == 0

    def redacted(self) -> dict:
        """Return a safe dict with no PII.

        Excludes the ``original_text`` and ``violations[].original`` fields.
        Use this method for logs, API responses, or database records::

            result = guard.scan(text)
            log.info("scan result: %s", result.redacted())

        Returns:
            A dict containing ``sanitized_text``, ``is_clean``, and violation
            metadata (entity_type, start, end, action, replacement).
            Raw PII values are not included.
        """
        return {
            "is_clean":       self.is_clean,
            "sanitized_text": self.sanitized_text,
            "scan_error":     self.scan_error,
            "violations": [
                {
                    "entity_type": v.entity_type,
                    "start":       v.start,
                    "end":         v.end,
                    "action":      v.action.value,
                    "replacement": v.replacement,
                    "confidence":  v.confidence,
                }
                for v in self.violations
            ],
        }

    def __repr__(self) -> str:
        return (
            f"ScanResult(is_clean={self.is_clean}, "
            f"violations={len(self.violations)})"
        )

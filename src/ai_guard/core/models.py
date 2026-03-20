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


# Known entity types — for typo checking and IDE support.
# A warning is issued if a type not in this list is configured.
KNOWN_ENTITY_TYPES: frozenset[str] = frozenset({
    "PERSON", "ORG", "EMAIL", "PHONE", "CREDIT_CARD", "IBAN",
    "TC_ID", "IP_ADDRESS", "IPv6", "ADDRESS", "POSTAL_CODE", "CUSTOM_SECRET",
    "UUID", "SSN", "MAC_ADDRESS", "JWT", "NIN",
    "UK_POSTAL_CODE", "US_ZIP_CODE", "EU_NATIONAL_ID", "PASSPORT",
    "CODICE_FISCALE",
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
            "violations": [
                {
                    "entity_type": v.entity_type,
                    "start":       v.start,
                    "end":         v.end,
                    "action":      v.action.value,
                    "replacement": v.replacement,
                }
                for v in self.violations
            ],
        }

    def __repr__(self) -> str:
        return (
            f"ScanResult(is_clean={self.is_clean}, "
            f"violations={len(self.violations)})"
        )

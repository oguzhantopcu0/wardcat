from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

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
    """Partially obscure the value, entity-aware. Most types reveal only the last
    few characters — e.g. ``CREDIT_CARD`` → ``************1111``,
    ``EMAIL`` → ``u***@example.com``, ``SSN`` → ``***-**-6789``. Types without a
    specific rule fall back to *first 2 + ``*`` + last 2* (``abcdef`` → ``ab**ef``);
    values shorter than 4 characters are fully replaced with ``*``.
    See ``ai_guard.core.engine._mask_value`` for the per-type rules."""


class Entity(str, Enum):
    """Known entity types, as constants — for autocomplete and typo-proofing.

    Use these instead of bare strings when configuring the guard::

        from ai_guard import Entity, Action

        guard.add_entity(Entity.CREDIT_CARD, action=Action.HASH)

    Each member *is* its string value (``Entity.EMAIL == "EMAIL"``), so it can be
    used anywhere a plain entity-type string is accepted. Note: because this is a
    ``(str, Enum)``, use ``.value`` to get the canonical string — ``str(Entity.EMAIL)``
    returns ``"Entity.EMAIL"``, not ``"EMAIL"``.

    The special member :attr:`Entity.All` is a sentinel, **not** a real entity
    type: passing it to :meth:`~ai_guard.AIGuard.add_entity` /
    :meth:`~ai_guard.AIGuard.remove_entity` enables/disables every known entity
    in one call. It is excluded from :data:`KNOWN_ENTITY_TYPES`.
    """

    All = "__ALL__"
    """Sentinel selecting *every* known entity type (not a real entity)."""

    PERSON = "PERSON"
    ORG = "ORG"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    CREDIT_CARD = "CREDIT_CARD"
    IBAN = "IBAN"
    TC_ID = "TC_ID"
    IP_ADDRESS = "IP_ADDRESS"
    IPv6 = "IPv6"
    ADDRESS = "ADDRESS"
    POSTAL_CODE = "POSTAL_CODE"
    CUSTOM_SECRET = "CUSTOM_SECRET"
    UUID = "UUID"
    SSN = "SSN"
    MAC_ADDRESS = "MAC_ADDRESS"
    JWT = "JWT"
    NIN = "NIN"
    UK_POSTAL_CODE = "UK_POSTAL_CODE"
    US_ZIP_CODE = "US_ZIP_CODE"
    EU_NATIONAL_ID = "EU_NATIONAL_ID"
    PASSPORT = "PASSPORT"
    CODICE_FISCALE = "CODICE_FISCALE"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    VEHICLE_PLATE = "VEHICLE_PLATE"
    FINANCIAL_AMOUNT = "FINANCIAL_AMOUNT"
    VAT_NUMBER = "VAT_NUMBER"
    SPECIAL_CATEGORY = "SPECIAL_CATEGORY"


# Known entity types — for typo checking and IDE support.
# Derived from Entity so the enum is the single source of truth (excluding the
# Entity.All sentinel): a warning is issued if a type not in this set is configured.
KNOWN_ENTITY_TYPES: frozenset[str] = frozenset(e.value for e in Entity if e is not Entity.All)


def warn_unknown_entity(entity_type: str) -> None:
    """Warn when an unknown entity type is used."""
    if entity_type not in KNOWN_ENTITY_TYPES:
        logger.warning(
            "Unknown entity type: %r — this type is not recognized. Typo? Valid types: %s",
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
    violations: list[Violation] = field(default_factory=list)
    """List of all detected violations. The ``original`` fields contain raw PII."""
    scan_error: str | None = None
    """Set when this item failed during :meth:`~ai_guard.AIGuard.scan_batch`.
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
            "is_clean": self.is_clean,
            "sanitized_text": self.sanitized_text,
            "scan_error": self.scan_error,
            "violations": [
                {
                    "entity_type": v.entity_type,
                    "start": v.start,
                    "end": v.end,
                    "action": v.action.value,
                    "replacement": v.replacement,
                    "confidence": v.confidence,
                }
                for v in self.violations
            ],
        }

    def __repr__(self) -> str:
        return f"ScanResult(is_clean={self.is_clean}, violations={len(self.violations)})"

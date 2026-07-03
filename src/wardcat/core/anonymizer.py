"""Anonymization stage — separate from detection.

The :class:`~wardcat.core.engine.DetectionEngine` *finds* spans (analysis); the
:class:`Anonymizer` *transforms* them (applies the configured action to each
span and rebuilds the text). Keeping the two apart mirrors the analyze →
anonymize split of mature PII pipelines and lets the action set be pluggable
(see :mod:`wardcat.core.actions`).
"""

from __future__ import annotations

from typing import Any

from wardcat.core.actions import ActionContext, get_action
from wardcat.core.models import Violation
from wardcat.detectors.base import DetectedSpan


class Anonymizer:
    """Applies configured actions to detected spans and rebuilds the text."""

    def __init__(self, entity_config: dict[str, Any], salt: str = "") -> None:
        self._entity_config = entity_config
        self._ctx = ActionContext(salt=salt)

    def apply(self, text: str, spans: list[DetectedSpan]) -> tuple[str, list[Violation]]:
        """Return ``(sanitized_text, violations)`` for *spans* (already filtered).

        *spans* must be sorted/non-overlapping (the engine guarantees this).
        """
        violations: list[Violation] = []
        sanitized = text
        offset = 0

        for span in spans:
            action_name = self._entity_config.get(span.entity_type, {}).get("action", "warn")
            replacement = get_action(action_name)(span, self._ctx)

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
                    action=action_name,
                    replacement=replacement,
                    confidence=span.confidence,
                )
            )

        return sanitized, violations

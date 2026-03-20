from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class DetectedSpan:
    """A single detected sensitive data span."""

    entity_type: str
    """Entity type — e.g. ``"EMAIL"``, ``"CREDIT_CARD"``."""
    text: str
    """Full text copied from the original."""
    start: int
    """Start index in the original text (inclusive)."""
    end: int
    """End index in the original text (exclusive)."""


class BaseDetector(ABC):
    """Interface that all detectors must implement."""

    @abstractmethod
    def detect(self, text: str) -> List[DetectedSpan]:
        """Scan text and return the list of found spans."""
        ...

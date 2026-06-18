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
    confidence: float = 1.0
    """Detection confidence in [0.0, 1.0]. Regex/checksum detections are 1.0;
    NER and LLM detections are 0.85 (model-based, not fully deterministic)."""


class BaseDetector(ABC):
    """Interface that all detectors must implement."""

    # True only for the LLM detector, which can adjudicate other detectors'
    # candidates. The engine routes candidates to detectors with this flag.
    can_adjudicate: bool = False

    @abstractmethod
    def detect(self, text: str) -> List[DetectedSpan]:
        """Scan text and return the list of found spans."""
        ...

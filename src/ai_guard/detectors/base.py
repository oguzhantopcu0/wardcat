from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass


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
    """Interface that all detectors implement.

    The engine talks to detectors only through this interface — it never imports
    a concrete detector. Two optional capabilities are expressed on the base so
    the engine stays decoupled:

    * ``can_adjudicate`` — when ``True`` the engine routes the other detectors'
      spans to this detector via the ``candidates`` argument (ensemble mode).
    * ``detect_async`` — a default thread-based implementation is provided;
      I/O-bound detectors (e.g. an LLM backend) override it with native async.
    """

    #: Set ``True`` on a detector that can verify/relabel/drop other detectors'
    #: candidate spans (currently the LLM detector). The engine reads this flag.
    can_adjudicate: bool = False

    @abstractmethod
    def detect(
        self, text: str, candidates: list[DetectedSpan] | None = None
    ) -> list[DetectedSpan]:
        """Scan *text* and return the spans found.

        :param candidates: spans found by the other detectors. Only meaningful for
            adjudicating detectors (``can_adjudicate=True``); others ignore it.
        """
        ...

    async def detect_async(
        self, text: str, candidates: list[DetectedSpan] | None = None
    ) -> list[DetectedSpan]:
        """Async variant of :meth:`detect`.

        Default implementation offloads the synchronous :meth:`detect` to a
        thread. I/O-bound detectors should override this with native async I/O.
        """
        return await asyncio.to_thread(self.detect, text, candidates)

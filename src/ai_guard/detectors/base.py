from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class DetectedSpan:
    """Tespit edilen tek bir hassas veri aralığı."""

    entity_type: str
    """Entity tipi — örn. ``"EMAIL"``, ``"CREDIT_CARD"``."""
    text: str
    """Orijinal metinden kopyalanmış tam metin."""
    start: int
    """Orijinal metindeki başlangıç indeksi (dahil)."""
    end: int
    """Orijinal metindeki bitiş indeksi (hariç)."""


class BaseDetector(ABC):
    """Tüm dedektörlerin uygulaması gereken arayüz."""

    @abstractmethod
    def detect(self, text: str) -> List[DetectedSpan]:
        """Metni tara ve bulunan span listesini döndür."""
        ...

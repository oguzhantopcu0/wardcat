from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List

logger = logging.getLogger(__name__)


class Action(str, Enum):
    """Tespit edilen PII üzerinde uygulanacak aksiyon."""

    WARN = "warn"
    """Metni olduğu gibi bırak, yalnızca ihlal olarak raporla."""
    HASH = "hash"
    """SHA-256 + salt ile maskele: ``[ENTITY_TYPE:abcd1234]``."""


# Bilinen entity tipleri — typo kontrolü ve IDE desteği için.
# Bu listeye olmayan bir tip configure edilirse Warning verilir.
KNOWN_ENTITY_TYPES: frozenset[str] = frozenset({
    "PERSON", "ORG", "EMAIL", "PHONE", "CREDIT_CARD", "IBAN",
    "TC_ID", "IP_ADDRESS", "IPv6", "ADDRESS", "POSTAL_CODE", "CUSTOM_SECRET",
    "UUID", "SSN", "MAC_ADDRESS", "JWT", "NIN",
})


def warn_unknown_entity(entity_type: str) -> None:
    """Bilinmeyen entity tipi kullanıldığında uyarı ver."""
    if entity_type not in KNOWN_ENTITY_TYPES:
        logger.warning(
            "Bilinmeyen entity tipi: %r — bu tip tanınmıyor. "
            "Yazım hatası mı? Geçerli tipler: %s",
            entity_type,
            sorted(KNOWN_ENTITY_TYPES),
        )


@dataclass
class Violation:
    """Metinde tespit edilen tek bir PII ihlali."""

    entity_type: str
    """Örn. ``"EMAIL"``, ``"CREDIT_CARD"``, ``"PERSON"``."""
    original: str
    """Orijinal metindeki ham değer."""
    start: int
    """Orijinal metindeki başlangıç indeksi."""
    end: int
    """Orijinal metindeki bitiş indeksi."""
    action: Action
    """Uygulanan aksiyon."""
    replacement: str | None = None
    """Hash aksiyonunda üretilen yer tutucu; warn'da ``None``."""


@dataclass
class ScanResult:
    """Tek bir ``guard.scan()`` çağrısının sonucu."""

    original_text: str
    """Değiştirilmemiş orijinal girdi."""
    sanitized_text: str
    """PII'ları maskelenmiş/raporlanmış çıktı metni."""
    violations: List[Violation] = field(default_factory=list)
    """Tespit edilen tüm ihlallerin listesi."""

    @property
    def is_clean(self) -> bool:
        """``True`` ise hiçbir PII tespit edilmedi."""
        return len(self.violations) == 0

    def __repr__(self) -> str:
        return (
            f"ScanResult(is_clean={self.is_clean}, "
            f"violations={len(self.violations)})"
        )

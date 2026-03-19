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
    "UK_POSTAL_CODE", "US_ZIP_CODE", "EU_NATIONAL_ID", "PASSPORT",
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
    """Tek bir ``guard.scan()`` çağrısının sonucu.

    .. warning::
        ``original_text`` ve ``violations[].original`` alanları ham PII içerir.
        Bu nesneyi log'a, veritabanına veya API yanıtına yazarken yalnızca
        ``sanitized_text`` kullanın. PII içermeyen bir dict için
        :meth:`redacted` metodunu kullanın.
    """

    original_text: str
    """Değiştirilmemiş orijinal girdi. **Ham PII içerir — dışarıya sızdırmayın.**"""
    sanitized_text: str
    """PII'ları maskelenmiş/raporlanmış çıktı metni."""
    violations: List[Violation] = field(default_factory=list)
    """Tespit edilen tüm ihlallerin listesi. ``original`` alanları ham PII içerir."""

    @property
    def is_clean(self) -> bool:
        """``True`` ise hiçbir PII tespit edilmedi."""
        return len(self.violations) == 0

    def redacted(self) -> dict:
        """PII içermeyen güvenli dict döndürür.

        ``original_text`` ve ``violations[].original`` alanlarını dışarıda bırakır.
        Log, API yanıtı veya veritabanı kaydı için bu metodu kullanın::

            result = guard.scan(text)
            log.info("scan result: %s", result.redacted())

        Returns:
            ``sanitized_text``, ``is_clean`` ve ihlal meta verilerini
            (entity_type, start, end, action, replacement) içeren dict.
            Ham PII değerleri dahil değildir.
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

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

from ai_guard.detectors.base import BaseDetector, DetectedSpan

# (pattern, flags) tuple — entity başına bayrak desteği
_SEP = r"[\s\-]?"   # kart numaralarında opsiyonel boşluk / tire

_PATTERNS: Dict[str, Tuple[str, int]] = {
    # ── Kredi kartı ──────────────────────────────────────────────────
    # Boşluklu ve bitişik formatları destekler: 4111111111111111 veya 4111 1111 1111 1111
    "CREDIT_CARD": (
        r"(?<!\d)"
        r"(?:"
        rf"4[0-9]{{3}}{_SEP}[0-9]{{4}}{_SEP}[0-9]{{4}}{_SEP}[0-9]{{4}}"   # Visa 16
        rf"|4[0-9]{{12}}"                                                    # Visa 13
        rf"|5[1-5][0-9]{{2}}{_SEP}[0-9]{{4}}{_SEP}[0-9]{{4}}{_SEP}[0-9]{{4}}"  # MasterCard
        rf"|3[47][0-9]{{2}}{_SEP}[0-9]{{6}}{_SEP}[0-9]{{5}}"              # Amex (4-6-5)
        rf"|3(?:0[0-5]|[68][0-9])[0-9]{{11}}"                              # Diners
        rf"|6(?:011|5[0-9]{{2}}){_SEP}[0-9]{{4}}{_SEP}[0-9]{{4}}{_SEP}[0-9]{{4}}"  # Discover
        r")"
        r"(?!\d)",
        0,
    ),
    # ── E-posta ──────────────────────────────────────────────────────
    "EMAIL": (
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        0,
    ),
    # ── Telefon ──────────────────────────────────────────────────────
    # Türk telefonu için 0 veya +90 ön eki zorunludur; salt 10 hane eşleşmez.
    "PHONE": (
        r"(?<!\d)"
        r"(?:"
        r"\+90[\s\-]?"              # +90 uluslararası
        r"|90[\s\-]?"               # 90 (başında + olmadan)
        r"|0"                       # ulusal 0 ön eki
        r")"
        r"(?:\(?\d{3}\)?)"          # alan kodu (parantezli veya değil)
        r"[\s\-]?\d{3}"
        r"[\s\-]?\d{2}"
        r"[\s\-]?\d{2}"
        r"(?!\d)",
        0,
    ),
    # ── IBAN ─────────────────────────────────────────────────────────
    # Büyük/küçük harf duyarsız (TR330006... veya tr330006... her ikisi de yakalanır)
    "IBAN": (
        r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]?\d{0,16})\b",
        re.IGNORECASE,
    ),
    # ── IP adresi ────────────────────────────────────────────────────
    "IP_ADDRESS": (
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
        0,
    ),
    # ── TC Kimlik No ─────────────────────────────────────────────────
    "TC_ID": (
        r"(?<!\d)[1-9][0-9]{10}(?!\d)",
        0,
    ),
    # ── Türkiye posta kodu: 01000–81999 ──────────────────────────────
    # Tire, harf veya rakam sonrası eşleşmeyi engeller (ürün kodu false positive'i önler)
    "POSTAL_CODE": (
        r"(?<![A-Za-zÇĞİÖŞÜçğışöşü0-9\-])(?:0[1-9]|[1-7]\d|80|81)\d{3}(?!\d)",
        0,
    ),
    # ── Türkçe adres ─────────────────────────────────────────────────
    # Terminatör olarak yalnızca gerçek adres anahtar kelimeleri kullanılır.
    # "No:" ve "Kat:" bağımsız terminatör olarak ÇIKARILDI: bu ifadeler
    # "TC Kimlik No:", "Kart No:" gibi kalıplarla çakışıp TC_ID ve
    # CREDIT_CARD detection'ını engelliyordu. Gerçek adreslerde No/Kat
    # zaten Cad., Sok., Mah. gibi bir keyword'ün ardından gelir.
    "ADDRESS": (
        r"(?:[A-ZÇĞİÖŞÜa-zçğışöşü0-9\.]+\s+){1,5}"
        r"(?:Mahallesi|Mah\.|Caddesi|Cad\.|Sokağı|Sokak|Sok\.|Bulvarı|Blv\."
        r"|Apartmanı|Apt\.|Sitesi)",
        0,
    ),
}

_COMPILED: Dict[str, re.Pattern] = {
    entity: re.compile(pattern, flags)
    for entity, (pattern, flags) in _PATTERNS.items()
}


class RegexDetector(BaseDetector):
    """Yapısal PII desenlerini regex ile tespit eder (CC, IBAN, TC_ID, e-posta, …)."""

    def __init__(self, enabled_entities: Set[str]) -> None:
        self.enabled_entities = enabled_entities

    def detect(self, text: str) -> List[DetectedSpan]:
        """Etkin entity tipleri için tüm regex eşleşmelerini döndür."""
        spans: List[DetectedSpan] = []
        for entity_type, pattern in _COMPILED.items():
            if entity_type not in self.enabled_entities:
                continue
            for match in pattern.finditer(text):
                spans.append(
                    DetectedSpan(
                        entity_type=entity_type,
                        text=match.group(),
                        start=match.start(),
                        end=match.end(),
                    )
                )
        return spans

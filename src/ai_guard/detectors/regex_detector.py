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
    # Uluslararası E.164 formatı da desteklenir (+1, +44, +49, vb.).
    "PHONE": (
        r"(?<!\d)"
        r"(?:"
        # Turkish: 0 / +90 / 90 prefix
        r"(?:\+90|90|0)[\s\-]?(?:\(?\d{3}\)?)[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"
        r"|"
        # International E.164 (non-Turkish): +1..., +44..., +49..., etc.
        r"\+(?!90)[1-9]\d{1,3}[\s\-]?\d{2,4}[\s\-]?\d{2,4}[\s\-]?\d{0,4}"
        r")"
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
    # Regex yalnızca format kontrolü yapar; checksum _validate_tc_id() ile doğrulanır.
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
    # ── Adres ────────────────────────────────────────────────────────
    # Türkçe adres: Mahallesi, Caddesi, Sokağı, vb. anahtar kelimeleriyle biter.
    # Uluslararası adres: numara + isim + Street/Avenue/Road/vb. formatı.
    # "No:" ve "Kat:" bağımsız terminatör olarak ÇIKARILDI: bu ifadeler
    # "TC Kimlik No:", "Kart No:" gibi kalıplarla çakışıp TC_ID ve
    # CREDIT_CARD detection'ını engelliyordu. Gerçek adreslerde No/Kat
    # zaten Cad., Sok., Mah. gibi bir keyword'ün ardından gelir.
    "ADDRESS": (
        # Turkish address patterns
        r"(?:[A-ZÇĞİÖŞÜa-zçğışöşü0-9\.]+\s+){1,5}"
        r"(?:Mahallesi|Mah\.|Caddesi|Cad\.|Sokağı|Sokak|Sok\.|Bulvarı|Blv\."
        r"|Apartmanı|Apt\.|Sitesi)"
        r"|"
        # International: street number + name + type keyword
        # {2,25} — 40 çok geniş, false positive üretiyordu; \b ile kesin sınır
        r"\b\d{1,5}[A-Za-z]?\s+[A-Za-z][A-Za-z\s\.]{2,25}"
        r"(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Boulevard|Blvd\.|Lane|Ln\."
        r"|Drive|Dr\.|Court|Ct\.|Way|Place|Pl\.|Square|Sq\.|Terrace|Terr\."
        r"|Close|Crescent|Gardens?|Highway|Hwy\.)\b",
        0,
    ),
    # ── Custom Secret ─────────────────────────────────────────────────
    # Bilinen token/credential prefix desenleri — net formatı olan servisler.
    # Bağlamsal (password=VALUE) tespiti LLM dedektörüne bırakılır.
    # Desteklenen:
    #   sk-...       — OpenAI / Anthropic API key
    #   ghp_/ghs_/gho_ — GitHub Personal/Server/OAuth token
    #   AKIA...      — AWS Access Key ID
    #   ya29.        — Google OAuth2 access token
    #   xoxb-/xoxp-  — Slack Bot/User token
    "CUSTOM_SECRET": (
        r"\b(?:"
        r"sk-[A-Za-z0-9_\-]{8,}"
        r"|ghp_[A-Za-z0-9]{20,}"
        r"|ghs_[A-Za-z0-9]{20,}"
        r"|gho_[A-Za-z0-9]{20,}"
        r"|AKIA[A-Z0-9]{16}"
        r"|ya29\.[A-Za-z0-9_\-]{20,}"
        r"|xoxb-[A-Za-z0-9\-]{20,}"
        r"|xoxp-[A-Za-z0-9\-]{30,}"
        r")",
        0,
    ),
    # ── UUID ─────────────────────────────────────────────────────────
    # RFC 4122 standard UUID: 8-4-4-4-12 hex digits
    "UUID": (
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        re.IGNORECASE,
    ),
    # ── SSN — US Social Security Number ──────────────────────────────
    # Format: 123-45-6789 (dashes required to avoid false positives)
    # Excludes invalid prefixes: 000, 666, 900-999
    "SSN": (
        r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
        0,
    ),
    # ── MAC Address ──────────────────────────────────────────────────
    # Colon or dash separated: 00:1A:2B:3C:4D:5E or 00-1A-2B-3C-4D-5E
    "MAC_ADDRESS": (
        r"\b(?:[0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}\b",
        0,
    ),
    # ── JWT — JSON Web Token ──────────────────────────────────────────
    # Always starts with eyJ (base64url of '{"') and has two dots
    "JWT": (
        r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]*",
        0,
    ),
    # ── IPv6 ─────────────────────────────────────────────────────────
    # Full form and compressed forms (::). Alternatives cover all RFC 5952 cases.
    "IPv6": (
        r"(?<![:\w])"
        r"(?:"
        r"(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}"                       # full
        r"|(?:[0-9a-fA-F]{1,4}:){1,7}:"                                    # trailing ::
        r"|:(?::[0-9a-fA-F]{1,4}){1,7}"                                    # leading ::
        r"|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}"                   # 1 gap
        r"|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}"         # 2 gap
        r"|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}"         # 3 gap
        r"|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}"         # 4 gap
        r"|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}"         # 5 gap
        r")"
        r"(?![:\w])",
        0,
    ),
    # ── NIN — UK National Insurance Number ───────────────────────────
    # Format: AB123456C — two prefix letters, 6 digits, one suffix A-D
    # Invalid prefixes excluded: D, F, I, Q, U, V as first/second letter
    "NIN": (
        r"\b[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]\b",
        re.IGNORECASE,
    ),
}

_COMPILED: Dict[str, re.Pattern] = {
    entity: re.compile(pattern, flags)
    for entity, (pattern, flags) in _PATTERNS.items()
}


def _validate_iban(value: str) -> bool:
    """IBAN mod-97 checksum doğrulaması (ISO 13616).

    Adımlar:
    1. Boşlukları kaldır, büyük harfe çevir.
    2. İlk 4 karakteri sona taşı.
    3. Her harfi sayıya çevir (A=10, …, Z=35).
    4. Sayısal dize % 97 == 1 ise geçerlidir.
    """
    cleaned = value.replace(" ", "").upper()
    if len(cleaned) < 5:
        return False
    rearranged = cleaned[4:] + cleaned[:4]
    numeric = ""
    for ch in rearranged:
        if ch.isdigit():
            numeric += ch
        elif ch.isalpha():
            numeric += str(ord(ch) - ord("A") + 10)
        else:
            return False
    try:
        return int(numeric) % 97 == 1
    except ValueError:
        return False


def _validate_tc_id(value: str) -> bool:
    """TC Kimlik No checksum doğrulaması (Türkiye Nüfus İdaresi algoritması).

    Kural:
    - d[0]..d[9] rakamları, d[10] kontrol rakamı.
    - (d[0]+d[2]+d[4]+d[6]+d[8]) * 7 - (d[1]+d[3]+d[5]+d[7]) mod 10 == d[9]
    - (d[0]+d[1]+...+d[9]) mod 10 == d[10]
    """
    if len(value) != 11 or not value.isdigit() or value[0] == "0":
        return False
    d = [int(c) for c in value]
    odd_sum  = d[0] + d[2] + d[4] + d[6] + d[8]
    even_sum = d[1] + d[3] + d[5] + d[7]
    if (odd_sum * 7 - even_sum) % 10 != d[9]:
        return False
    if sum(d[:10]) % 10 != d[10]:
        return False
    return True


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
                value = match.group()
                # Checksum doğrulamaları — false positive baskısı
                if entity_type == "TC_ID" and not _validate_tc_id(value):
                    continue
                if entity_type == "IBAN" and not _validate_iban(value):
                    continue
                spans.append(
                    DetectedSpan(
                        entity_type=entity_type,
                        text=value,
                        start=match.start(),
                        end=match.end(),
                    )
                )
        return spans

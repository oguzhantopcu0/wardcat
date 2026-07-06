from __future__ import annotations

import concurrent.futures
import logging
import re
from collections.abc import Callable

from wardcat.detectors.base import BaseDetector, DetectedSpan
from wardcat.utils.normalize import fold_confusables, has_confusables

logger = logging.getLogger(__name__)

# (pattern, flags) tuple — per-entity flag support
_SEP = r"[ \-\.]{0,2}"  # optional space/dash/dot in card numbers (up to 2 chars)

_PATTERNS: dict[str, tuple[str, int]] = {
    # ── Credit card ──────────────────────────────────────────────────
    # Supports spaced and compact formats: 4111111111111111 or 4111 1111 1111 1111
    "CREDIT_CARD": (
        r"(?<!\d)"
        r"(?:"
        rf"4[0-9]{{3}}{_SEP}[0-9]{{4}}{_SEP}[0-9]{{4}}{_SEP}[0-9]{{4}}"  # Visa 16
        rf"|4[0-9]{{12}}"  # Visa 13
        rf"|5[1-5][0-9]{{2}}{_SEP}[0-9]{{4}}{_SEP}[0-9]{{4}}{_SEP}[0-9]{{4}}"  # MasterCard
        rf"|3[47][0-9]{{2}}{_SEP}[0-9]{{6}}{_SEP}[0-9]{{5}}"  # Amex (4-6-5)
        rf"|3(?:0[0-5]|[68][0-9])[0-9]{{11}}"  # Diners
        rf"|6(?:011|5[0-9]{{2}}){_SEP}[0-9]{{4}}{_SEP}[0-9]{{4}}{_SEP}[0-9]{{4}}"  # Discover
        r")"
        r"(?!\d)",
        0,
    ),
    # ── Email ──────────────────────────────────────────────────────────
    "EMAIL": (
        r"\b[\w._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        0,
    ),
    # ── Phone ──────────────────────────────────────────────────────────
    # Turkish phone requires 0 or +90 prefix; bare 10 digits will not match.
    # International E.164 format is also supported (+1, +44, +49, etc.).
    "PHONE": (
        r"(?<!\d)"
        r"(?:"
        # Turkish: 0 / +90 / 90 prefix
        r"(?:\+90|90|0)[\s\-]?(?:\(?\d{3}\)?)[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"
        r"|"
        # French national: 0X followed by 4 pairs, e.g. 01 23 45 67 89 / 01.23.45.67.89
        r"0[1-9](?:[\s.]\d{2}){4}"
        r"|"
        # German mobile: 015x/016x/017x + 6-8 digits, e.g. 0151 23456789
        r"01[5-7]\d[\s/\-]?\d{6,8}"
        r"|"
        # International E.164 (non-Turkish): +1..., +44..., +49..., etc.
        r"\+(?!90)[1-9]\d{1,3}[\s\-]?\d{2,4}[\s\-]?\d{2,4}[\s\-]?\d{0,4}"
        r")"
        r"(?!\d)",
        0,
    ),
    # ── IBAN ─────────────────────────────────────────────────────────
    # Supports both compact (TR330006...) and spaced (TR33 0006 1005...) formats.
    # Groups of 4 alphanumeric characters, optionally separated by single spaces.
    # Minimum BBAN: 2 full groups (e.g. Norway 15 chars); maximum: 8 groups + remainder.
    # Checksum is validated by _validate_iban() which strips spaces before mod-97.
    "IBAN": (
        r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,8}(?:[ ]?[A-Z0-9]{1,4})?\b",
        re.IGNORECASE,
    ),
    # ── IP address ────────────────────────────────────────────────────
    "IP_ADDRESS": (
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
        0,
    ),
    # ── Turkish National ID (TC Kimlik No) ────────────────────────────
    # Regex performs format check only; checksum validated by _validate_tc_id().
    "TC_ID": (
        r"(?<!\d)[1-9][0-9]{10}(?!\d)",
        0,
    ),
    # ── Turkey postal code: 01000–81999 ──────────────────────────────
    # Prevents matching after a dash, letter, or digit (avoids product code false positives)
    "POSTAL_CODE": (
        r"(?<![A-Za-zÇĞİÖŞÜçğışöşü0-9\-])(?:0[1-9]|[1-7]\d|80|81)\d{3}(?!\d)",
        0,
    ),
    # ── Address ────────────────────────────────────────────────────────
    # Turkish: Mahallesi, Caddesi, Sokağı, etc.
    # English: number + name + street type (Street, Road, etc.)
    # French: Rue, Allée, Impasse, Promenade, Place + name
    # Spanish: Calle, Avenida, Plaza, Paseo, Carrera + name
    # Italian: Viale, Piazza, Corso, Vicolo, Largo + name
    # Dutch: compound street names (straat, weg, gracht, etc.)
    # German: compound street names (straße, gasse, weg, platz, etc.)
    "ADDRESS": (
        # Turkish: 1-3 *capitalized* words before the street-type keyword (so it
        # can't swallow lowercase filler like "iletişime geçilebilir adresi" or
        # cross a sentence boundary), then an optional No:/Daire:/Kat: tail.
        r"(?:[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğışöşü0-9]*\s+){1,3}"
        r"(?:Mahallesi|Mah\.|Caddesi|Cad\.|Sokağı|Sokak|Sok\.|Bulvarı|Blv\."
        r"|Apartmanı|Apt\.|Sitesi)"
        r"(?:\s+No[:.]?\s*\d+[A-Za-z]?)?"
        r"(?:\s+(?:Kat|Daire|Blok|D)[:.]?\s*\d+)?"
        r"|"
        # English/international: number + name + street type keyword
        r"\b\d{1,5}[A-Za-z]?\s+[A-Za-z][A-Za-z\s\.]{2,25}"
        r"(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Boulevard|Blvd\.|Lane|Ln\."
        r"|Drive|Dr\.|Court|Ct\.|Way|Place|Pl\.|Square|Sq\.|Terrace|Terr\."
        r"|Close|Crescent|Gardens?|Highway|Hwy\.)\b"
        r"|"
        # French: Rue/Allée/Impasse/Promenade/Place + optional article + proper name
        r"\b(?:Rue|All[eé]e|Impasse|Promenade|Place)\s+"
        r"(?:de\s+(?:la\s+|l'|les?\s+)?|du\s+|des\s+)?"
        r"[A-ZÀ-Ö][A-Za-zÀ-ÖØ-öø-ÿ\-]{2,20}(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ\-]{2,15}){0,3}\b"
        r"|"
        # Spanish: Calle/Avenida/Plaza/Paseo/Carrera + optional article + name
        r"\b(?:Calle|Avenida|Plaza|Paseo|Carrera)\s+"
        r"(?:de\s+(?:la\s+|los?\s+|las?\s+)?|del\s+)?"
        r"[A-ZÁÉÍÓÚÑ][A-Za-záéíóúñ\-]{2,20}(?:\s+[A-Za-záéíóúñ\-]{2,15}){0,3}\b"
        r"|"
        # Italian: Viale/Piazza/Corso/Vicolo/Largo + optional article + name
        r"\b(?:Viale|Piazza|Corso|Vicolo|Largo)\s+"
        r"(?:della?\s+|dello?\s+|dei?\s+|del\s+|delle\s+|degli\s+)?"
        r"[A-ZÀ-Ö][A-Za-zÀ-ÖØ-öø-ÿ\-]{2,20}(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ\-]{2,15}){0,3}\b"
        r"|"
        # Dutch compound streets: Kalverstraat, Keizersgracht, Prinsenlaan + optional number
        r"\b[A-Z][a-z]{2,20}(?:straat|weg|laan|plein|gracht|kade|dijk)\b"
        r"(?:\s+\d{1,5}[a-z]?)?"
        r"|"
        # German compound streets: Hauptstraße 15, Musterweg 7, Lindenallee
        r"\b[A-ZÄÖÜ][a-zäöüß]{2,20}"
        r"(?:stra(?:ss|ß)e|gasse|weg|platz|ring|allee|damm|ufer|chaussee)\b"
        r"(?:\s+\d{1,5}[a-z]?)?",
        0,
    ),
    # ── UK Postal Code ──────────────────────────────────────────────────
    # Format: AA9A 9AA, AA99 9AA, AA9 9AA, A9 9AA, A9A 9AA, A99 9AA
    # Examples: SW1A 1AA, EC1A 1BB, W1A 1HQ, M1 1AE, GU21 6TH
    "UK_POSTAL_CODE": (
        r"\b[A-Z]{1,2}[0-9][0-9A-Z]?\s?[0-9][A-Z]{2}\b",
        re.IGNORECASE,
    ),
    # ── US ZIP Postal Code ─────────────────────────────────────────────
    # ZIP+4: 12345-6789 (lowest false positive rate)
    # Labeled: "ZIP: 90210" or "zip code: 10001" (explicit context)
    "US_ZIP_CODE": (
        r"\b\d{5}-\d{4}\b"
        r"|"
        # Labeled bare ZIP: "ZIP: 90210". The negative lookahead prevents this
        # branch from grabbing only the first 5 digits of a ZIP+4 (90210-1234),
        # which would otherwise leak the "-1234" suffix.
        r"\b[Zz][Ii][Pp](?:\s*[Cc][Oo][Dd][Ee])?\s*:?\s*\d{5}(?!-\d{4})\b",
        0,
    ),
    # ── EU National Identity Number ────────────────────────────────────
    # Spain DNI: 8 digits + check letter (TRWAGMYFPDXBNJZSQVHLCKE)
    # Spain NIE (foreigners): X/Y/Z + 7 digits + check letter
    # France INSEE (social security): gender(1) + year(2) + month(01-12) + 9 digits = 15 digits
    "EU_NATIONAL_ID": (
        r"\b\d{8}[TRWAGMYFPDXBNJZSQVHLCKE]\b"
        r"|"
        r"\b[XYZ]\d{7}[TRWAGMYFPDXBNJZSQVHLCKE]\b"
        r"|"
        r"\b[12]\d{2}(?:0[1-9]|1[0-2])\d{10}\b",
        0,
    ),
    # ── Passport Number ───────────────────────────────────────────────────
    # Context-aware: only matches when preceded by a passport keyword.
    # Supports: English, Turkish, German, French, Spanish.
    # Format: 1-2 capital letters + 6-9 digits (covers most countries).
    "PASSPORT": (
        r"\bpassport\s*(?:no\.?|number|num\.?|#)?\s*:?\s*[A-Z]{1,2}\d{6,9}\b"
        r"|"
        r"\bpasaport\s*(?:no\.?|numaras[iı])?\s*:?\s*[A-Z]{1,2}\d{6,9}\b"
        r"|"
        r"\breisepass\s*(?:nr\.?|nummer)?\s*:?\s*[A-Z]{1,2}\d{6,9}\b"
        r"|"
        r"\bpasseport\s*(?:no\.?|num[eé]ro)?\s*:?\s*[A-Z]{1,2}\d{6,9}\b"
        r"|"
        r"\bpasaporte\s*(?:no\.?|n[uú]mero)?\s*:?\s*[A-Z]{1,2}\d{6,9}\b",
        re.IGNORECASE,
    ),
    # ── Italian Codice Fiscale ─────────────────────────────────────────────
    # Italian personal tax code: 6 letters + 2 digits + letter + 2 digits + letter + 3 digits + letter
    # Example: RSSMRA85T10A562S — 16 chars, very distinctive pattern.
    "CODICE_FISCALE": (
        r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b",
        re.IGNORECASE,
    ),
    # ── Date of Birth ─────────────────────────────────────────────────
    # Two detection strategies to balance recall vs. false positives:
    #
    # 1. Month-name formats (DD Month YYYY) — low false-positive risk;
    #    matched without requiring a keyword prefix.
    #    e.g.  "15 Mart 1988"  "3 January 2001"
    #
    # 2. Numeric / ISO formats (DD.MM.YYYY, YYYY-MM-DD) — high false-positive
    #    risk (any date can appear in this format); matched ONLY when preceded
    #    by an explicit birth-date keyword.
    #    e.g.  "doğum tarihi: 15.03.1988"  "date of birth: 1988-03-15"
    #          "born: 01/05/1992"          "dob: 1990-07-22"
    "DATE_OF_BIRTH": (
        # Strategy 1: DD <MonthName> YYYY — only plausible birth years (1900–2015),
        # no keyword required. Years 2016+ require a keyword to avoid false positives
        # on future/effective calendar dates (e.g. "15 Haziran 2025 itibarıyla").
        r"\b\d{1,2}\.?\s+"
        # Turkish
        r"(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık"
        # English
        r"|January|February|March|April|May|June|July|August|September|October|November|December"
        r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
        # German
        r"|Januar|Februar|März|Mai|Juni|Juli|Oktober|Dezember|Mär|Okt|Dez"
        # French
        r"|janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre"
        r"|févr|avr|juil|sept|déc)"
        r"\s+(?:19\d{2}|20(?:0\d|1[0-5]))\b"
        r"|"
        # Strategy 2: numeric/ISO formats — keyword required (any year)
        # Keywords: TR, EN, German (Geburtsdatum/geboren am/geb.),
        # French (date de naissance / né(e) le)
        r"(?:doğum(?:\s+tarihi)?|d\.?t\.?|born(?:\s+on)?|date\s+of\s+birth|d\.?o\.?b\.?|birthday"
        r"|geburtsdatum|geboren\s+am|geb\.?"
        r"|date\s+de\s+naissance|n[ée]e?\s+le)\s*:?\s*"
        r"(?:\d{1,2}[./]\d{1,2}[./](?:19|20)\d{2}|(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01]))",
        re.IGNORECASE,
    ),
    # ── Financial Amount ─────────────────────────────────────────────────
    # Monetary amounts with explicit currency symbol or unit.
    # Matches: ₺47.3 milyon, $2.1 milyon, 85.000 TL, €500.000
    # Requires a currency marker to avoid false positives on bare numbers.
    "FINANCIAL_AMOUNT": (
        # Currency symbol first: ₺47.3 milyon / $2.1 / €500.000
        r"(?:₺|\$|€|£)\s*\d+(?:[.,]\d+)*(?:\s*(?:milyon|milyar|bin|million|billion|trillion|thousand))?"
        r"|"
        # Amount then TL/lira: 85.000 TL / 47.3 milyon TL
        r"\b\d+(?:[.,]\d+)*(?:\s*(?:milyon|milyar|bin))?\s*(?:TL|lira)\b",
        re.IGNORECASE,
    ),
    # ── VAT / Tax Number ──────────────────────────────────────────────
    # Country-prefixed EU VAT numbers (distinctive) + context-required
    # Turkish tax number (Vergi No / VKN — 10 digits, ambiguous without a keyword).
    # DE: DE123456789 · FR: FRXX999999999 · GB: GB999999999(999) ·
    # IT: IT12345678901 · ES: ESX9999999X · AT: ATU99999999 · NL: NL999999999B99
    "VAT_NUMBER": (
        r"\b(?:"
        r"DE\d{9}"
        r"|FR[A-Z0-9]{2}\d{9}"
        r"|GB\d{9}(?:\d{3})?"
        r"|IT\d{11}"
        r"|ES[A-Z0-9]\d{7}[A-Z0-9]"
        r"|ATU\d{8}"
        r"|NL\d{9}B\d{2}"
        r")\b"
        r"|(?i:vergi\s*(?:kimlik\s*)?no|vkn)\s*:?\s*\d{10}\b",
        0,
    ),
    # ── Custom Secret ─────────────────────────────────────────────────
    # Known token/credential prefix patterns — services with well-defined formats.
    # Contextual detection (password=VALUE) is delegated to the LLM detector.
    # Supported:
    #   sk-... / sk-ant-...   — OpenAI / Anthropic API key
    #   sk_live_/sk_test_/rk_live_ — Stripe secret/restricted key
    #   ghp_/ghs_/gho_        — GitHub Personal/Server/OAuth token
    #   glpat-                — GitLab personal access token
    #   AKIA...               — AWS Access Key ID
    #   AIza...               — Google API key
    #   ya29.                 — Google OAuth2 access token
    #   xoxb-/xoxp-           — Slack Bot/User token
    #   SG....                — SendGrid API key
    #   SK<32hex>/AC<32hex>   — Twilio API key / Account SID
    #   npm_...               — npm access token
    #   hooks.slack.com/...   — Slack incoming webhook URL
    #   -----BEGIN ... PRIVATE KEY----- — PEM private key block
    "CUSTOM_SECRET": (
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
        r"[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
        r"|https://hooks\.slack\.com/services/[A-Za-z0-9/_\-]{20,}"
        r"|\b(?:"
        r"sk-ant-[A-Za-z0-9_\-]{16,}"
        r"|sk-[A-Za-z0-9_\-]{8,}"
        r"|(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{10,}"
        r"|ghp_[A-Za-z0-9]{20,}"
        r"|ghs_[A-Za-z0-9]{20,}"
        r"|gho_[A-Za-z0-9]{20,}"
        r"|glpat-[A-Za-z0-9_\-]{20,}"
        r"|AKIA[A-Z0-9]{16}"
        r"|AIza[A-Za-z0-9_\-]{35}"
        r"|ya29\.[A-Za-z0-9_\-]{20,}"
        r"|xoxb-[A-Za-z0-9\-]{20,}"
        r"|xoxp-[A-Za-z0-9\-]{30,}"
        r"|SG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}"
        r"|SK[0-9a-fA-F]{32}"
        r"|AC[0-9a-fA-F]{32}"
        r"|npm_[A-Za-z0-9]{36}"
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
        r"(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}"  # full
        r"|(?:[0-9a-fA-F]{1,4}:){1,7}:"  # trailing ::
        r"|:(?::[0-9a-fA-F]{1,4}){1,7}"  # leading ::
        r"|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}"  # 1 gap
        r"|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}"  # 2 gap
        r"|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}"  # 3 gap
        r"|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}"  # 4 gap
        r"|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}"  # 5 gap
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
    # ── Turkish Vehicle Plate ─────────────────────────────────────────
    # Format: <city_code> <letters> <digits>
    #   city_code: 01–81 (2 digits, Turkish province codes)
    #   letters:   1–3 uppercase Latin letters (A-Z, no Turkish special chars)
    #   digits:    2–4 digits
    # Examples: "34 ABC 123", "06 AZ 1234", "81 T 4321", "34ABC123"
    # Spaced and compact forms are both matched.
    "VEHICLE_PLATE": (
        r"(?<!\d)"
        r"(?:0[1-9]|[1-7]\d|80|81)"  # city code 01–81
        r"[\s]?"
        r"[A-Z]{1,3}"  # 1–3 Latin letters
        r"[\s]?"
        r"\d{2,4}"  # 2–4 digits
        r"(?!\d)",
        0,
    ),
}

_COMPILED: dict[str, re.Pattern] = {
    entity: re.compile(pattern, flags) for entity, (pattern, flags) in _PATTERNS.items()
}

# ── Connection-string credentials ────────────────────────────────────────────
# URI userinfo with a password: scheme://user:password@host
# e.g. postgresql://admin:Sup3rS3cr3t@db.prod.internal:5432/appdb
# The password is a CUSTOM_SECRET. Without this, the EMAIL pattern greedily
# matches "password@host" as an email and, being the longer span, wins overlap
# resolution — leaving the real secret undetected. We capture the password and
# suppress the spurious EMAIL match that starts at the same offset.
_URI_CREDENTIAL: re.Pattern = re.compile(
    r"[a-z][a-z0-9+.\-]*://[^\s:/@]*:(?P<pwd>[^\s@/]+)@",
    re.IGNORECASE,
)

# Timeout for custom pattern execution (seconds). Built-in patterns are trusted.
_CUSTOM_PATTERN_TIMEOUT = 2.0


def _safe_finditer(pattern: re.Pattern, text: str) -> list:
    """Execute a regex pattern with a timeout to prevent ReDoS at runtime.

    Returns a list of match objects, or an empty list if the pattern times out.
    Used only for user-supplied custom patterns; built-in patterns are trusted.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(list, pattern.finditer(text))
        try:
            return future.result(timeout=_CUSTOM_PATTERN_TIMEOUT)
        except concurrent.futures.TimeoutError:
            logger.warning(
                "Custom pattern %r timed out after %.1fs on input of length %d — skipped.",
                pattern.pattern,
                _CUSTOM_PATTERN_TIMEOUT,
                len(text),
            )
            return []


# ISO 3166-1 alpha-2 country codes that have published IBAN formats (SWIFT/IBAN registry).
_VALID_IBAN_COUNTRIES: frozenset[str] = frozenset(
    {
        "AD",
        "AE",
        "AL",
        "AT",
        "AZ",
        "BA",
        "BE",
        "BG",
        "BH",
        "BR",
        "BY",
        "CH",
        "CR",
        "CY",
        "CZ",
        "DE",
        "DK",
        "DO",
        "EE",
        "EG",
        "ES",
        "FI",
        "FO",
        "FR",
        "GB",
        "GE",
        "GI",
        "GL",
        "GR",
        "GT",
        "HR",
        "HU",
        "IE",
        "IL",
        "IQ",
        "IS",
        "IT",
        "JO",
        "KW",
        "KZ",
        "LB",
        "LC",
        "LI",
        "LT",
        "LU",
        "LV",
        "LY",
        "MC",
        "MD",
        "ME",
        "MK",
        "MR",
        "MT",
        "MU",
        "NL",
        "NO",
        "PK",
        "PL",
        "PS",
        "PT",
        "QA",
        "RO",
        "RS",
        "SA",
        "SC",
        "SD",
        "SE",
        "SI",
        "SK",
        "SM",
        "ST",
        "SV",
        "TL",
        "TN",
        "TR",
        "UA",
        "VA",
        "VG",
        "XK",
    }
)


def _validate_iban(value: str) -> bool:
    """IBAN mod-97 checksum validation (ISO 13616).

    Steps:
    1. Remove spaces, convert to uppercase.
    2. Move first 4 characters to the end.
    3. Convert each letter to a number (A=10, …, Z=35).
    4. Valid if numeric string % 97 == 1.
    """
    cleaned = value.replace(" ", "").upper()
    if len(cleaned) < 5:
        return False
    if cleaned[:2] not in _VALID_IBAN_COUNTRIES:
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


def _validate_luhn(value: str) -> bool:
    """Luhn (mod-10) checksum validation for credit/debit card numbers.

    Strips non-digits, then applies the standard Luhn algorithm. Rejects
    format-valid but mathematically impossible card numbers, eliminating a
    large class of false positives (random 16-digit sequences).
    """
    digits = [int(c) for c in value if c.isdigit()]
    if len(digits) < 12:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _validate_tc_id(value: str) -> bool:
    """TC National ID checksum validation (Turkish Civil Registration algorithm).

    Rules:
    - d[0]..d[9] are digits, d[10] is the check digit.
    - (d[0]+d[2]+d[4]+d[6]+d[8]) * 7 - (d[1]+d[3]+d[5]+d[7]) mod 10 == d[9]
    - (d[0]+d[1]+...+d[9]) mod 10 == d[10]
    """
    if len(value) != 11 or not value.isdigit() or value[0] == "0":
        return False
    d = [int(c) for c in value]
    odd_sum = d[0] + d[2] + d[4] + d[6] + d[8]
    even_sum = d[1] + d[3] + d[5] + d[7]
    if (odd_sum * 7 - even_sum) % 10 != d[9]:
        return False
    if sum(d[:10]) % 10 != d[10]:
        return False
    return True


# Entity-type → checksum/structural validator. A regex match is only accepted
# as a violation when its validator (if any) returns True. Adding a new
# validated entity is a one-line registry entry — no changes to detect().
_VALIDATORS: dict[str, Callable[[str], bool]] = {
    "TC_ID": _validate_tc_id,
    "IBAN": _validate_iban,
    "CREDIT_CARD": _validate_luhn,
}

# ── Confidence tiers ──────────────────────────────────────────────────────────
# Not all regex matches are equally certain. Tiering the confidence lets the
# engine keep a proven match while letting a fuzzy one be overridden by the LLM
# adjudicator. Checksum > high-precision structural > fuzzy > (model layers 0.85).
CONF_CHECKSUM = 1.0  # mathematically verified — TC_ID / IBAN / CREDIT_CARD
CONF_STRUCTURAL = 0.97  # distinctive, high-precision format — email, JWT, IP, secrets…
CONF_FUZZY = 0.90  # distinctive-but-ambiguous — can over/under-match

# Entities whose regex is inherently fuzzy (keyword/heuristic, not a rigid format).
_FUZZY_ENTITIES: frozenset[str] = frozenset({"ADDRESS", "VEHICLE_PLATE"})


def _regex_confidence(entity_type: str) -> float:
    if entity_type in _VALIDATORS:
        return CONF_CHECKSUM
    if entity_type in _FUZZY_ENTITIES:
        return CONF_FUZZY
    return CONF_STRUCTURAL


class RegexDetector(BaseDetector):
    """Detects structural PII patterns using regex (CC, IBAN, TC_ID, email, …)."""

    def __init__(
        self,
        enabled_entities: set[str],
        custom_patterns: dict | None = None,
        *,
        fold_confusables_enabled: bool = True,
    ) -> None:
        self.enabled_entities = enabled_entities
        # When True, matching runs on a confusable-folded copy of the input so
        # homoglyph-obfuscated PII (Cyrillic/Greek lookalikes, fullwidth/Arabic
        # digits) is still detected. Folding is length-preserving, so spans are
        # reported against the original text. See wardcat.utils.normalize.
        self._fold_confusables = fold_confusables_enabled
        custom_patterns = custom_patterns or {}
        # Compile custom patterns: {entity_type: (compiled_pattern, action)}
        self._custom_compiled: dict[str, tuple[re.Pattern, str]] = {}
        for name, cfg in custom_patterns.items():
            pattern_str = cfg.get("pattern", "")
            action = cfg.get("action", "warn")
            try:
                self._custom_compiled[name] = (re.compile(pattern_str), action)
            except re.error as exc:
                logger.warning("Custom pattern %r could not be compiled: %s — skipped.", name, exc)

    def detect(self, text: str, candidates: list[DetectedSpan] | None = None) -> list[DetectedSpan]:
        """Return all regex matches for enabled entity types."""
        spans: list[DetectedSpan] = []

        # Run matching on a confusable-folded copy so homoglyph-obfuscated PII is
        # caught. Folding is length-preserving (see fold_confusables), so a match
        # at [start, end) in `scan_text` covers the same slice of `text`; we report
        # the *original* substring (what redaction must remove) and validate the
        # *folded* value (the canonical ASCII form the checksum expects).
        scan_text = (
            fold_confusables(text) if self._fold_confusables and has_confusables(text) else text
        )

        # Connection-string credentials: capture the password as a secret and
        # mark its offset so the spurious EMAIL match (password@host) is dropped.
        cred_pwd_starts: set[int] = set()
        for match in _URI_CREDENTIAL.finditer(scan_text):
            cred_pwd_starts.add(match.start("pwd"))
            if "CUSTOM_SECRET" in self.enabled_entities:
                spans.append(
                    DetectedSpan(
                        entity_type="CUSTOM_SECRET",
                        text=text[match.start("pwd") : match.end("pwd")],
                        start=match.start("pwd"),
                        end=match.end("pwd"),
                    )
                )

        for entity_type, pattern in _COMPILED.items():
            if entity_type not in self.enabled_entities:
                continue
            for match in pattern.finditer(scan_text):
                folded_value = match.group()
                original_value = text[match.start() : match.end()]
                # Drop EMAIL matches that are actually a URI credential's
                # "password@host" (the password is captured as CUSTOM_SECRET).
                if entity_type == "EMAIL" and match.start() in cred_pwd_starts:
                    continue
                # Checksum/structural validation — suppress false positives.
                # Validate the folded (canonical) form so a homoglyph digit does
                # not defeat the checksum.
                validator = _VALIDATORS.get(entity_type)
                if validator is not None and not validator(folded_value):
                    logger.debug(
                        "%s format match rejected (failed validation): %r — "
                        "if this is real PII, the value may be incorrectly formatted.",
                        entity_type,
                        folded_value,
                    )
                    continue
                spans.append(
                    DetectedSpan(
                        entity_type=entity_type,
                        text=original_value,
                        start=match.start(),
                        end=match.end(),
                    )
                )
        # Custom patterns — no checksum validation; use timeout wrapper for ReDoS safety
        for entity_type, (pattern, _action) in self._custom_compiled.items():
            for match in _safe_finditer(pattern, text):
                value = match.group()
                spans.append(
                    DetectedSpan(
                        entity_type=entity_type,
                        text=value,
                        start=match.start(),
                        end=match.end(),
                    )
                )
        # Tier the confidence by how certain the match is, so overlap resolution
        # and LLM adjudication can treat a fuzzy match differently from a proven
        # one (a checksum span is never overridable; a fuzzy ADDRESS is).
        for s in spans:
            s.confidence = _regex_confidence(s.entity_type)
        return spans

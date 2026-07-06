"""
Unicode confusable ("homoglyph") folding — an offset-preserving normalizer.

A common way to slip PII past an ASCII-oriented regex is to swap a Latin
character for a visually identical one from another script: ``ali@tеst.com``
uses a Cyrillic ``е`` (U+0435) in the domain, so the ASCII-only domain class
never matches and the email is missed. Fullwidth (``４111…``) and Arabic-Indic
(``٤111…``) digits do the same to numeric patterns (cards, IBANs, IDs).

:func:`fold_confusables` maps each such character to its ASCII "skeleton"
**one code point at a time**, so the folded string has exactly the same length
as the input. That invariant is what lets a detector run its regexes on the
folded copy yet report spans by the *original* text's offsets — no offset
remapping needed.

The map is deliberately curated (not the full Unicode confusables table): only
unambiguous, same-case Latin lookalikes plus digit/fullwidth ranges. NFKC is
*not* used because its compatibility decompositions can change string length
(e.g. the ``ﬁ`` ligature → ``fi``), which would break the offset invariant.
"""

from __future__ import annotations

# ── Curated confusable → ASCII map ────────────────────────────────────────────
# Cyrillic letters that are visually identical to a Latin letter of the SAME case.
# (Case-ambiguous ones like Cyrillic "р" vs Latin "p" are kept; genuinely
# different-shape letters are omitted to avoid corrupting legitimate text.)
_CYRILLIC: dict[str, str] = {
    # lowercase
    "а": "a",
    "е": "e",
    "о": "o",
    "р": "p",
    "с": "c",
    "х": "x",
    "у": "y",
    "і": "i",
    "ј": "j",
    "ѕ": "s",
    "ԁ": "d",
    "ԛ": "q",
    "ԝ": "w",
    "ё": "e",
    # uppercase
    "А": "A",
    "В": "B",
    "Е": "E",
    "К": "K",
    "М": "M",
    "Н": "H",
    "О": "O",
    "Р": "P",
    "С": "C",
    "Т": "T",
    "У": "Y",
    "Х": "X",
    "І": "I",
    "Ј": "J",
    "Ѕ": "S",
    "Ё": "E",
}

# Greek letters visually identical to a Latin letter of the same case.
_GREEK: dict[str, str] = {
    # lowercase
    "ο": "o",
    "α": "a",
    "ν": "v",
    "ρ": "p",
    "ι": "i",
    # uppercase
    "Α": "A",
    "Β": "B",
    "Ε": "E",
    "Ζ": "Z",
    "Η": "H",
    "Ι": "I",
    "Κ": "K",
    "Μ": "M",
    "Ν": "N",
    "Ο": "O",
    "Ρ": "P",
    "Τ": "T",
    "Υ": "Y",
    "Χ": "X",
}


def _digit_range(start: int) -> dict[str, str]:
    """Map a contiguous block of 10 digit code points to ASCII ``0``–``9``."""
    return {chr(start + i): str(i) for i in range(10)}


def _fullwidth_letters() -> dict[str, str]:
    """Map fullwidth Latin letters (U+FF21–FF3A, U+FF41–FF5A) to ASCII."""
    upper = {chr(0xFF21 + i): chr(ord("A") + i) for i in range(26)}
    lower = {chr(0xFF41 + i): chr(ord("a") + i) for i in range(26)}
    return {**upper, **lower}


# Digit lookalikes: Arabic-Indic (U+0660), Extended Arabic-Indic (U+06F0),
# and fullwidth (U+FF10) — all decimal digits, all length-preserving.
_DIGITS: dict[str, str] = {
    **_digit_range(0x0660),
    **_digit_range(0x06F0),
    **_digit_range(0xFF10),
}

# Single str.translate table (ordinal → single-char string). Every value is one
# character, so ``str.translate`` is guaranteed length-preserving.
_TABLE: dict[int, str] = {
    ord(k): v for k, v in {**_CYRILLIC, **_GREEK, **_DIGITS, **_fullwidth_letters()}.items()
}


def fold_confusables(text: str) -> str:
    """Fold Unicode confusables to their ASCII skeleton, preserving length.

    The returned string has the same length as ``text`` and identical code-point
    offsets, so a match found in the folded string can be reported using the
    original text's ``start``/``end`` (and the original substring). Characters
    without a mapping are left unchanged.
    """
    return text.translate(_TABLE)


def has_confusables(text: str) -> bool:
    """True if ``text`` contains at least one character the folder would rewrite.

    Lets callers skip the folded-copy detection pass entirely for the common
    all-ASCII case.
    """
    return any(ord(ch) in _TABLE for ch in text)

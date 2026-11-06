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

import unicodedata

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
# Fullwidth commercial at: ``ali＠example.com`` is an email to a reader.
_TABLE[0xFF20] = "@"


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


# ── Invisible characters ──────────────────────────────────────────────────────
# Folding cannot delete a character, so a zero-width space inside a card number
# survives it and splits the number in two. Matching therefore runs on a
# *prepared* copy that drops such characters, with an index map back to the
# original so spans still cover the original text — the invisible characters
# inside a value included.

# Not format characters (Cf), but invisible all the same: the combining
# grapheme joiner and the variation selectors.
_DROP_EXTRA = frozenset({0x034F, *range(0xFE00, 0xFE10), *range(0xE0100, 0xE01F0)})


def _prepared_char(ch: str) -> str | None:
    """What *ch* becomes in the prepared copy: ``None`` drops it."""
    code = ord(ch)
    if code < 0x80:
        return ch
    if code in _DROP_EXTRA:
        return None
    category = unicodedata.category(ch)
    if category == "Cf":
        return None
    if category == "Zs":
        return " "
    if category == "Nd":
        return str(unicodedata.decimal(ch))
    return ch


def needs_matching_prep(text: str) -> bool:
    """True if :func:`prepare_for_matching` would change *text*.

    Lets callers keep the direct path for the common case. Plain ASCII, and
    text whose only non-ASCII characters are letters (Turkish, say), needs
    nothing.
    """
    if text.isascii():
        return False
    return any(ord(ch) >= 0x80 and _prepared_char(ch) != ch for ch in text)


def prepare_for_matching(text: str) -> tuple[str, list[int]]:
    """Return a copy of *text* fit for pattern matching, and an index map.

    Format characters (Unicode category ``Cf``: zero-width space and joiners,
    the BOM, soft hyphen, bidi controls) and variation selectors are dropped;
    every space separator (``Zs``: NBSP, thin and narrow spaces) becomes a plain
    space; every decimal digit (``Nd``: Devanagari, mathematical, …) becomes its
    ASCII digit. ``index_map[i]`` is the position in *text* of the prepared
    copy's ``i``-th character, so a match ``[s, e)`` in the copy covers
    ``[index_map[s], index_map[e - 1] + 1)`` in the original.

    Confusable folding is separate (:func:`fold_confusables`); it preserves
    length and can run on the prepared copy afterwards.
    """
    chars: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(text):
        out = _prepared_char(ch)
        if out is not None:
            chars.append(out)
            index_map.append(i)
    return "".join(chars), index_map


def to_original_span(start: int, end: int, index_map: list[int]) -> tuple[int, int]:
    """Map a ``[start, end)`` span of a prepared copy back to the original."""
    if end <= start:
        position = index_map[start] if start < len(index_map) else index_map[-1] + 1
        return position, position
    return index_map[start], index_map[end - 1] + 1

"""Text helpers shared by the detectors and the sensitivity gate."""

from __future__ import annotations

import re

_PARA_RE = re.compile(r"\n+")

# A suffix written after an apostrophe at the end of a name: Turkish case and
# possessive endings ("Yılmaz'ın", "İstanbul'da", "Telekom'un") and the English
# possessive ("John's"). The part before it must be at least two letters, which
# keeps a name whose apostrophe belongs to it: "o'brien" is not cut to "o". The
# suffix must be lower case and contain a vowel (or be a bare "s"), so "O'Brien"
# and "D'Angelo" never match. Some models also swallow the first letter or two
# of the next word ("Yılmaz’a e" from "Yılmaz’a e-posta"); that tail goes too.
_NAME_SUFFIX = re.compile(
    r"(?<=[^\W\d_]{2})['’](?:s|[a-zçğıöşü]*[aeıioöuü][a-zçğıöşü]*)(?:\s+[a-zçğıöşü]{1,2})?$"
)
_NAME_SUFFIX_MAX = 12


def strip_name_suffix(value: str, start: int, end: int) -> tuple[str, int, int]:
    """Cut an apostrophe suffix off the end of a detected name, adjusting ``end``.

    Turkish attaches case endings to proper nouns with an apostrophe, and NER
    models include the ending in the entity. Left there, one person becomes a
    different value in every grammatical case — "Ahmet Yılmaz", "Ahmet Yılmaz'ın",
    "Ahmet Yılmaz'a" — so each gets its own hash and an index cannot tell they are
    the same person. Replacing only the name keeps the value stable and leaves
    the ending in the text: ``[PERSON:…]'ın``.
    """
    match = _NAME_SUFFIX.search(value)
    if match is None or len(match.group()) > _NAME_SUFFIX_MAX:
        return value, start, end
    cut = match.start()
    return value[:cut], start, end - (len(value) - cut)


def chunk_by_paragraph(text: str, max_chars: int) -> list[tuple[str, int]]:
    """Split *text* into ``(chunk, start_offset)`` pairs at paragraph boundaries.

    Paragraphs are grouped greedily so each chunk stays within ``max_chars``.
    Keeps each LLM call within an attention-friendly size and stops long inputs
    from being silently truncated. Returns a single ``(text, 0)`` chunk when
    ``max_chars <= 0`` or the text already fits.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return [(text, 0)]

    # Paragraph segments (spans of non-separator text between newline runs).
    segs: list[tuple[int, int]] = []
    pos = 0
    for m in _PARA_RE.finditer(text):
        if m.start() > pos:
            segs.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        segs.append((pos, len(text)))
    if not segs:
        return [(text, 0)]

    result: list[tuple[str, int]] = []
    chunk_start, chunk_end = segs[0]
    for seg_start, seg_end in segs[1:]:
        if seg_end - chunk_start > max_chars:
            result.append((text[chunk_start:chunk_end], chunk_start))
            chunk_start = seg_start
        chunk_end = seg_end
    result.append((text[chunk_start:chunk_end], chunk_start))
    return result

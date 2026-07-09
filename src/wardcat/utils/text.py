"""Text chunking shared by the LLM detector and the sensitivity gate."""

from __future__ import annotations

import re

_PARA_RE = re.compile(r"\n+")


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

"""Reverse of the anonymization stage — put the real values back.

:class:`~wardcat.core.anonymizer.Anonymizer` replaces detected spans with
placeholders; this module walks a piece of text (typically an LLM's answer to a
sanitized prompt), swaps every placeholder back for the value it stood for, and
reports what it did as an ordered, citation-style source list.

Reversal is driven purely by the ``replacement`` recorded on each
:class:`~wardcat.core.models.Violation`, so it works for whichever action
produced the text — but only ``tokenize`` guarantees a unique placeholder per
value. ``hash`` is unique in practice (a salted digest per distinct value);
``redact`` and ``mask`` collapse different values onto the same placeholder
(two names both become ``[PERSON]``), and a placeholder that stood for more than
one value is reported as ``ambiguous`` and left untouched rather than guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from wardcat.core.models import Violation

#: Why a detected value was not put back into the text.
UnrestoredReason = Literal["not-present", "ambiguous", "not-replaced"]

_SOURCES_TITLE = "Sources"


@dataclass(frozen=True)
class Substitution:
    """One placeholder that was swapped back for its original value.

    .. warning::
        ``original`` is raw PII — the whole point of restoring — so a
        :class:`RestoredText` is as sensitive as the input it came from.
    """

    index: int
    """1-based position in the source list, ordered by first appearance in the text."""
    entity_type: str
    """E.g. ``"EMAIL"``, ``"PERSON"``."""
    action: str
    """Action that produced the placeholder (``"tokenize"``, ``"hash"``, …)."""
    placeholder: str
    """The text that stood in for the value (``"[PERSON_1]"``)."""
    original: str
    """The value that was put back. **Raw PII.**"""
    occurrences: int
    """How many times the placeholder appeared in the restored text."""
    confidence: float
    """Detection confidence of the underlying violation."""


@dataclass(frozen=True)
class UnrestoredValue:
    """A detected value that was *not* put back, and why.

    Carries no raw value: ``not-present`` is the common, boring case (the answer
    simply never mentioned that placeholder), and the caller can look the value
    up in ``violations`` when it matters.
    """

    entity_type: str
    action: str
    placeholder: str | None
    """``None`` for report-only actions such as ``warn``, which replace nothing."""
    reason: UnrestoredReason
    """``"not-present"`` — the placeholder does not occur in the text (an empty
    replacement, which a custom action may return, occurs nowhere locatable);
    ``"ambiguous"`` — it stood for more than one distinct value, so restoring it
    would be a guess; ``"not-replaced"`` — the action left the original text in
    place (``warn``), so there is nothing to reverse."""


@dataclass(frozen=True)
class RestoredText:
    """Result of :meth:`~wardcat.ScanResult.restore` — the text plus its sources.

    ``str(restored)`` is the text with the source list appended, which is usually
    what you want to show::

        answer = call_llm(result.sanitized_text)
        print(result.restore(answer))

    .. warning::
        Contains raw PII in ``text`` and in every ``substitutions[].original``.
    """

    text: str
    """The text with every unambiguous placeholder replaced by its original value."""
    substitutions: list[Substitution] = field(default_factory=list)
    """What was put back, ordered by first appearance in :attr:`text`."""
    unrestored: list[UnrestoredValue] = field(default_factory=list)
    """Detected values that were not put back, with the reason for each."""

    @property
    def is_complete(self) -> bool:
        """``True`` when nothing was skipped for being ambiguous.

        A ``not-present`` entry does not make a restore incomplete — an answer
        that never mentions a placeholder has nothing to put back.
        """
        return not any(u.reason == "ambiguous" for u in self.unrestored)

    def sources_block(self, *, title: str = _SOURCES_TITLE, notes: bool = True) -> str:
        """Render the ordered source list — which filter fired, and what it hid.

        ::

            [1] [PERSON_1] → Ali Veli (PERSON · tokenize · x2 · confidence 0.85)
            [2] [EMAIL_1] → ali@example.com (EMAIL · tokenize · confidence 0.97)

        Returns ``""`` when this text carries nothing to cite — nothing was put
        back and no placeholder was left in it — so appending the block to a
        clean answer adds nothing. A value that was masked but never mentioned in
        the text is not, by itself, something to cite.

        :param title: heading for the block.
        :param notes: also summarize the values that were *not* put back.
        """
        if not self.substitutions and self.is_complete:
            return ""

        lines = [f"--- {title} ---"]
        for sub in self.substitutions:
            facts = [sub.entity_type, sub.action]
            if sub.occurrences > 1:
                facts.append(f"x{sub.occurrences}")
            facts.append(f"confidence {sub.confidence:.2f}")
            lines.append(f"[{sub.index}] {sub.placeholder} → {sub.original} ({' · '.join(facts)})")

        if notes:
            lines.extend(self._notes())
        return "\n".join(lines) if len(lines) > 1 else ""

    def with_sources(self, *, title: str = _SOURCES_TITLE, notes: bool = True) -> str:
        """:attr:`text` with :meth:`sources_block` appended below it."""
        block = self.sources_block(title=title, notes=notes)
        return f"{self.text}\n\n{block}" if block else self.text

    def _notes(self) -> list[str]:
        """One trailing line per reason, summarizing what was left alone."""
        notes = []
        ambiguous = [u for u in self.unrestored if u.reason == "ambiguous"]
        absent = [u for u in self.unrestored if u.reason == "not-present"]
        if ambiguous:
            placeholders = ", ".join(sorted({u.placeholder or "" for u in ambiguous}))
            notes.append(
                f"(left in place: {placeholders} — each stood for more than one value, "
                "so restoring it would be a guess. Use the tokenize action to reverse "
                "these too.)"
            )
        if absent:
            types = ", ".join(sorted({u.entity_type for u in absent}))
            notes.append(f"({len(absent)} masked value(s) not mentioned here: {types}.)")
        return notes

    def __str__(self) -> str:
        return self.with_sources()

    def __repr__(self) -> str:
        return (
            f"RestoredText(substitutions={len(self.substitutions)}, "
            f"unrestored={len(self.unrestored)}, is_complete={self.is_complete})"
        )


def restore_text(text: str, violations: list[Violation]) -> RestoredText:
    """Put the originals recorded in *violations* back into *text*.

    Every placeholder is matched literally and all its occurrences are replaced in
    a single left-to-right pass, so a restored value can never be re-matched by
    another placeholder. Longer placeholders are tried first, so one that is a
    prefix of another (``ab**ef`` inside ``ab**efgh``) cannot shadow it.
    """
    by_placeholder: dict[str, list[Violation]] = {}
    unrestored: list[UnrestoredValue] = []

    for violation in violations:
        if violation.replacement is None:
            unrestored.append(
                UnrestoredValue(
                    entity_type=violation.entity_type,
                    action=violation.action,
                    placeholder=None,
                    reason="not-replaced",
                )
            )
            continue
        by_placeholder.setdefault(violation.replacement, []).append(violation)

    restorable: dict[str, Violation] = {}
    for placeholder, group in by_placeholder.items():
        first = group[0]
        if len({v.original for v in group}) > 1:
            unrestored.append(
                UnrestoredValue(
                    entity_type=first.entity_type,
                    action=first.action,
                    placeholder=placeholder,
                    reason="ambiguous",
                )
            )
            continue
        if not placeholder or placeholder not in text:
            unrestored.append(
                UnrestoredValue(
                    entity_type=first.entity_type,
                    action=first.action,
                    placeholder=placeholder,
                    reason="not-present",
                )
            )
            continue
        restorable[placeholder] = first

    if not restorable:
        return RestoredText(text=text, substitutions=[], unrestored=unrestored)

    # Longest first: a placeholder that is a prefix of another must not win.
    pattern = re.compile("|".join(re.escape(p) for p in sorted(restorable, key=len, reverse=True)))
    counts: dict[str, int] = {}
    order: list[str] = []

    def _swap(match: re.Match[str]) -> str:
        placeholder = match.group(0)
        if placeholder not in counts:
            order.append(placeholder)
            counts[placeholder] = 0
        counts[placeholder] += 1
        return restorable[placeholder].original

    restored = pattern.sub(_swap, text)

    substitutions = [
        Substitution(
            index=i,
            entity_type=restorable[placeholder].entity_type,
            action=restorable[placeholder].action,
            placeholder=placeholder,
            original=restorable[placeholder].original,
            occurrences=counts[placeholder],
            confidence=restorable[placeholder].confidence,
        )
        for i, placeholder in enumerate(order, start=1)
    ]
    return RestoredText(text=restored, substitutions=substitutions, unrestored=unrestored)

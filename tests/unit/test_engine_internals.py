"""
DetectionEngine internal mechanism tests.

Scope:
  - _resolve_overlaps: all overlap geometries
  - scan() offset tracking: text integrity after multiple consecutive replacements
  - Violation ordering: compliance with order of appearance in text
  - Empty input behavior
"""

from __future__ import annotations

from ai_guard.core.engine import DetectionEngine
from ai_guard.detectors.base import BaseDetector, DetectedSpan

# ── Helper: fake detector that returns a fixed span list ────────────────────


class _FixedDetector(BaseDetector):
    def __init__(self, spans: list[DetectedSpan]) -> None:
        self._spans = spans

    def detect(self, text: str) -> list[DetectedSpan]:
        return list(self._spans)


def _engine(spans: list[DetectedSpan], entity_actions: dict[str, str]) -> DetectionEngine:
    config = {
        "salt": "test",
        "entities": {k: {"enabled": True, "action": v} for k, v in entity_actions.items()},
    }
    return DetectionEngine(config, [_FixedDetector(spans)])


# ═══════════════════════════════════════════════════════════════════════════
# _resolve_overlaps
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveOverlaps:
    def _engine(self):
        return DetectionEngine({"salt": "", "entities": {}}, [])

    def _span(self, t, s, e, text="x"):
        return DetectedSpan(t, text[s:e] if len(text) > e else "x", s, e)

    # Basic geometries
    def test_empty_input(self):
        assert self._engine()._resolve_overlaps([]) == []

    def test_single_span(self):
        span = DetectedSpan("A", "abc", 0, 3)
        result = self._engine()._resolve_overlaps([span])
        assert result == [span]

    def test_no_overlap_adjacent(self):
        spans = [DetectedSpan("A", "ab", 0, 2), DetectedSpan("B", "cd", 2, 4)]
        result = self._engine()._resolve_overlaps(spans)
        assert len(result) == 2

    def test_no_overlap_with_gap(self):
        spans = [DetectedSpan("A", "a", 0, 1), DetectedSpan("B", "b", 5, 6)]
        result = self._engine()._resolve_overlaps(spans)
        assert len(result) == 2

    # Overlap — longer wins
    def test_overlap_longer_wins(self):
        long = DetectedSpan("LONG", "hello world", 0, 11)
        short = DetectedSpan("SHORT", "hello", 0, 5)
        result = self._engine()._resolve_overlaps([long, short])
        assert len(result) == 1
        assert result[0].entity_type == "LONG"

    def test_overlap_same_start_first_if_longer(self):
        a = DetectedSpan("A", "abcde", 0, 5)
        b = DetectedSpan("B", "abc", 0, 3)
        result = self._engine()._resolve_overlaps(sorted([a, b], key=lambda s: s.start))
        assert result[0].entity_type == "A"

    def test_overlap_contained_span_discarded(self):
        outer = DetectedSpan("OUTER", "hello world", 0, 11)
        inner = DetectedSpan("INNER", "world", 6, 11)
        result = self._engine()._resolve_overlaps([outer, inner])
        assert len(result) == 1
        assert result[0].entity_type == "OUTER"

    def test_overlap_shorter_first_in_list_but_longer_wins(self):
        """The one that appears first in ordering but is shorter loses."""
        short = DetectedSpan("SHORT", "abc", 0, 3)
        long = DetectedSpan("LONG", "abcdef", 0, 6)
        # resolve_overlaps receives sorted input (by start); same start for both (0)
        result = self._engine()._resolve_overlaps(sorted([short, long], key=lambda s: s.start))
        assert result[0].entity_type == "LONG"

    def test_three_way_overlap_longest_wins(self):
        spans = [
            DetectedSpan("S", "ab", 0, 2),
            DetectedSpan("M", "abcde", 0, 5),
            DetectedSpan("L", "abcdefg", 0, 7),
        ]
        result = self._engine()._resolve_overlaps(sorted(spans, key=lambda s: s.start))
        assert len(result) == 1
        assert result[0].entity_type == "L"

    def test_non_overlapping_after_overlap_chain(self):
        """The span following an overlap should also be preserved."""
        spans = [
            DetectedSpan("A", "hello", 0, 5),
            DetectedSpan("B", "hel", 0, 3),  # overlaps A, A wins
            DetectedSpan("C", "world", 6, 11),  # no overlap
        ]
        result = self._engine()._resolve_overlaps(sorted(spans, key=lambda s: s.start))
        assert len(result) == 2
        types = {s.entity_type for s in result}
        assert "A" in types
        assert "C" in types

    # Confidence priority — a checksum/regex span beats a longer fuzzy one
    def test_higher_confidence_beats_longer(self):
        """A high-confidence (regex/checksum) span wins over a longer NER/LLM span."""
        card = DetectedSpan("CREDIT_CARD", "4532015112830366", 0, 16, confidence=1.0)
        addr = DetectedSpan("ADDRESS", "4532015112830366 Main St", 0, 24, confidence=0.85)
        result = self._engine()._resolve_overlaps([addr, card])
        assert len(result) == 1
        assert result[0].entity_type == "CREDIT_CARD"

    def test_equal_confidence_falls_back_to_length(self):
        """With equal confidence the longer span still wins."""
        long = DetectedSpan("L", "x", 0, 11, confidence=0.85)
        short = DetectedSpan("S", "x", 0, 5, confidence=0.85)
        result = self._engine()._resolve_overlaps([short, long])
        assert result[0].entity_type == "L"

    # Chained / nested overlaps — every candidate is checked against all kept spans
    def test_chained_partial_overlaps_no_span_slips_through(self):
        """The winner's neighbours drop even when they don't overlap each other.

        A[0,4) and C[7,11) do not overlap one another, but both overlap the
        longest span B[2,9). A naive resolver that only compares against the
        previously kept span could let one of them slip through; checking every
        kept span drops both.
        """
        spans = [
            DetectedSpan("A", "x", 0, 4, confidence=1.0),
            DetectedSpan("B", "x", 2, 9, confidence=1.0),  # longest → wins
            DetectedSpan("C", "x", 7, 11, confidence=1.0),
        ]
        result = self._engine()._resolve_overlaps(sorted(spans, key=lambda s: s.start))
        assert len(result) == 1
        assert result[0].entity_type == "B"

    def test_partial_overlap_does_not_leak_via_replacement(self):
        """Two spans overlapping only at their edge: the weaker one is dropped, not
        silently replaced in a way that resurrects a third overlapping span."""
        # High-confidence B[3,6) sits inside a low-confidence A[0,10). Even though A
        # is longer, B's higher confidence wins; the leftover A region does not
        # smuggle a second span back in.
        a = DetectedSpan("A", "x", 0, 10, confidence=0.85)
        b = DetectedSpan("B", "x", 3, 6, confidence=1.0)
        result = self._engine()._resolve_overlaps(sorted([a, b], key=lambda s: s.start))
        assert len(result) == 1
        assert result[0].entity_type == "B"

    def test_result_sorted_by_start(self):
        """Output is always sorted by start regardless of ranking order."""
        spans = [
            DetectedSpan("C", "x", 20, 25, confidence=1.0),
            DetectedSpan("A", "x", 0, 5, confidence=0.85),
            DetectedSpan("B", "x", 10, 15, confidence=0.85),
        ]
        result = self._engine()._resolve_overlaps(spans)
        assert [s.entity_type for s in result] == ["A", "B", "C"]
        assert [s.start for s in result] == sorted(s.start for s in result)


# ═══════════════════════════════════════════════════════════════════════════
# Offset tracking — text integrity after multiple replacements
# ═══════════════════════════════════════════════════════════════════════════


class TestOffsetTracking:
    """
    Verifies that when multiple hash replacements are applied,
    subsequent replacements are computed at the correct positions.
    """

    def _scan_with_known_spans(
        self, text: str, spans: list[DetectedSpan], actions: dict[str, str]
    ) -> tuple:
        engine = _engine(spans, actions)
        result = engine.scan(text)
        return result

    def test_single_replacement_correctness(self):
        text = "prefix [TARGET] suffix"
        spans = [DetectedSpan("X", "TARGET", 8, 14)]
        result = _engine(spans, {"X": "hash"}).scan(text)
        assert "TARGET" not in result.sanitized_text
        assert "[X:" in result.sanitized_text
        assert result.sanitized_text.startswith("prefix ")
        assert result.sanitized_text.endswith(" suffix")

    def test_two_replacements_order_preserved(self):
        text = "A: FIRST | B: SECOND"
        spans = [
            DetectedSpan("TYPE1", "FIRST", 3, 8),
            DetectedSpan("TYPE2", "SECOND", 14, 20),
        ]
        result = _engine(spans, {"TYPE1": "hash", "TYPE2": "hash"}).scan(text)
        assert "FIRST" not in result.sanitized_text
        assert "SECOND" not in result.sanitized_text
        assert "[TYPE1:" in result.sanitized_text
        assert "[TYPE2:" in result.sanitized_text
        # TYPE1 placeholder should appear before TYPE2 placeholder
        assert result.sanitized_text.index("[TYPE1:") < result.sanitized_text.index("[TYPE2:")

    def test_five_replacements_all_correct(self):
        """All 5 consecutive entities should be replaced at the correct position."""
        # Use values different from entity type names; type name appears in placeholder
        text = "a:CARD b:TCNO c:MAIL d:PHONE e:IBAN_NUM"

        # Compute positions from the text — eliminates manual error risk
        def _pos(word):
            s = text.index(word)
            return s, s + len(word)

        spans = [
            DetectedSpan("CC", "CARD", *_pos("CARD")),
            DetectedSpan("TC", "TCNO", *_pos("TCNO")),
            DetectedSpan("EM", "MAIL", *_pos("MAIL")),
            DetectedSpan("PH", "PHONE", *_pos("PHONE")),
            DetectedSpan("IB", "IBAN_NUM", *_pos("IBAN_NUM")),
        ]
        actions = dict.fromkeys(["CC", "TC", "EM", "PH", "IB"], "hash")
        result = _engine(spans, actions).scan(text)

        # Original content values should not be in the sanitized text
        for original in ["CARD", "TCNO", "MAIL", "PHONE", "IBAN_NUM"]:
            assert original not in result.sanitized_text, (
                f"'{original}' still present in sanitized text: {result.sanitized_text!r}"
            )

        # All placeholders should be present in the text
        for t in ["CC", "TC", "EM", "PH", "IB"]:
            assert f"[{t}:" in result.sanitized_text

        # Fixed text sections (a:, b:, etc.) should be preserved
        for label in ["a:", "b:", "c:", "d:", "e:"]:
            assert label in result.sanitized_text

    def test_warn_does_not_shift_offset(self):
        """warn action should not shift offset; subsequent hash should be at the correct position."""
        text = "W: WARN | H: HASH"
        spans = [
            DetectedSpan("W", "WARN", 3, 7),
            DetectedSpan("H", "HASH", 13, 17),
        ]
        result = _engine(spans, {"W": "warn", "H": "hash"}).scan(text)
        assert "WARN" in result.sanitized_text  # warn → unchanged
        assert "HASH" not in result.sanitized_text
        assert "[H:" in result.sanitized_text

    def test_replacement_longer_than_original(self):
        """If the replacement is longer than the original, the next span should still be in the correct position."""
        text = "A: X | B: Y"
        # 'X' → '[VERY_LONG_TYPE_NAME:abcd1234]' (much longer)
        spans = [
            DetectedSpan("VERY_LONG_TYPE_NAME", "X", 3, 4),
            DetectedSpan("B", "Y", 10, 11),
        ]
        result = _engine(spans, {"VERY_LONG_TYPE_NAME": "hash", "B": "hash"}).scan(text)
        assert "[VERY_LONG_TYPE_NAME:" in result.sanitized_text
        assert "[B:" in result.sanitized_text

    def test_replacement_shorter_than_original(self):
        """Even if the replacement is shorter than the original, the next span should still be in the correct position."""
        text = "AAAAAAAAAA | B: Y"
        spans = [
            DetectedSpan("LONG", "AAAAAAAAAA", 0, 10),  # 10 char → ~22 char placeholder
            DetectedSpan("B", "Y", 15, 16),
        ]
        result = _engine(spans, {"LONG": "hash", "B": "hash"}).scan(text)
        assert "[LONG:" in result.sanitized_text
        assert "[B:" in result.sanitized_text


# ═══════════════════════════════════════════════════════════════════════════
# Violation ordering
# ═══════════════════════════════════════════════════════════════════════════


class TestViolationOrdering:
    def test_violations_in_text_order(self):
        """Violations should be returned in order of appearance in the text."""
        text = "first: ALPHA second: BETA third: GAMMA"
        spans = [
            DetectedSpan("C", "GAMMA", 33, 38),
            DetectedSpan("A", "ALPHA", 7, 12),
            DetectedSpan("B", "BETA", 21, 25),
        ]
        result = _engine(spans, {"A": "warn", "B": "warn", "C": "warn"}).scan(text)
        entity_order = [v.entity_type for v in result.violations]
        assert entity_order == ["A", "B", "C"]

    def test_violation_start_positions_ascending(self):
        from ai_guard import AIGuard

        guard = AIGuard(use_ner=False)
        text = "a@a.com 0532 111 22 33 b@b.com"
        result = guard.scan(text)
        starts = [v.start for v in result.violations]
        assert starts == sorted(starts)


# ═══════════════════════════════════════════════════════════════════════════
# Empty / minimal input
# ═══════════════════════════════════════════════════════════════════════════


class TestEmptyAndMinimalInput:
    def test_empty_string(self):
        result = _engine([], {}).scan("")
        assert result.is_clean
        assert result.sanitized_text == ""
        assert result.original_text == ""

    def test_whitespace_only(self):
        result = _engine([], {}).scan("   \n\t  ")
        assert result.is_clean

    def test_single_char(self):
        result = _engine([], {}).scan("x")
        assert result.is_clean

    def test_no_detectors_means_clean(self):
        engine = DetectionEngine({"salt": "", "entities": {}}, [])
        result = engine.scan("kart: 4111111111111111 TC: 12345678950")
        assert result.is_clean

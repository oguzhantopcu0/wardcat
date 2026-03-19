"""
DetectionEngine iç mekanizması testleri.

Kapsam:
  - _resolve_overlaps: tüm çakışma geometrileri
  - scan() offset takibi: çoklu ardışık replacement sonrası metin bütünlüğü
  - Violation sıralama: metindeki görünüm sırasına uygunluk
  - Boş girdi davranışı
"""
from __future__ import annotations

import pytest

from ai_guard.core.engine import DetectionEngine
from ai_guard.core.models import Action
from ai_guard.detectors.base import BaseDetector, DetectedSpan


# ── Yardımcı: sabit span listesi döndüren sahte dedektör ────────────────────

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

    # Temel geometriler
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

    # Çakışma — uzunluk kazanır
    def test_overlap_longer_wins(self):
        long  = DetectedSpan("LONG",  "hello world", 0, 11)
        short = DetectedSpan("SHORT", "hello",        0,  5)
        result = self._engine()._resolve_overlaps([long, short])
        assert len(result) == 1
        assert result[0].entity_type == "LONG"

    def test_overlap_same_start_first_if_longer(self):
        a = DetectedSpan("A", "abcde", 0, 5)
        b = DetectedSpan("B", "abc",   0, 3)
        result = self._engine()._resolve_overlaps(sorted([a, b], key=lambda s: s.start))
        assert result[0].entity_type == "A"

    def test_overlap_contained_span_discarded(self):
        outer = DetectedSpan("OUTER", "hello world", 0, 11)
        inner = DetectedSpan("INNER", "world",        6, 11)
        result = self._engine()._resolve_overlaps([outer, inner])
        assert len(result) == 1
        assert result[0].entity_type == "OUTER"

    def test_overlap_shorter_first_in_list_but_longer_wins(self):
        """Sıralamada önce gelen ama daha kısa olan kaybeder."""
        short = DetectedSpan("SHORT", "abc",    0, 3)
        long  = DetectedSpan("LONG",  "abcdef", 0, 6)
        # resolve_overlaps girişi sıralı (start'a göre) alır; aynı start için her ikisi de 0
        result = self._engine()._resolve_overlaps(sorted([short, long], key=lambda s: s.start))
        assert result[0].entity_type == "LONG"

    def test_three_way_overlap_longest_wins(self):
        spans = [
            DetectedSpan("S", "ab",     0, 2),
            DetectedSpan("M", "abcde",  0, 5),
            DetectedSpan("L", "abcdefg",0, 7),
        ]
        result = self._engine()._resolve_overlaps(sorted(spans, key=lambda s: s.start))
        assert len(result) == 1
        assert result[0].entity_type == "L"

    def test_non_overlapping_after_overlap_chain(self):
        """Çakışma sonrası gelen span da korunmalı."""
        spans = [
            DetectedSpan("A", "hello", 0, 5),
            DetectedSpan("B", "hel",   0, 3),   # A ile çakışır, A kazanır
            DetectedSpan("C", "world", 6, 11),  # çakışmaz
        ]
        result = self._engine()._resolve_overlaps(sorted(spans, key=lambda s: s.start))
        assert len(result) == 2
        types = {s.entity_type for s in result}
        assert "A" in types
        assert "C" in types


# ═══════════════════════════════════════════════════════════════════════════
# Offset takibi — çoklu replacement sonrası metin bütünlüğü
# ═══════════════════════════════════════════════════════════════════════════

class TestOffsetTracking:
    """
    Birden fazla hash replacement yapıldığında, sonraki replacement'ların
    konumlarının doğru hesaplandığını doğrular.
    """

    def _scan_with_known_spans(self, text: str, spans: list[DetectedSpan],
                                actions: dict[str, str]) -> tuple:
        engine = _engine(spans, actions)
        result = engine.scan(text)
        return result

    def test_single_replacement_correctness(self):
        text  = "prefix [TARGET] suffix"
        spans = [DetectedSpan("X", "TARGET", 8, 14)]
        result = _engine(spans, {"X": "hash"}).scan(text)
        assert "TARGET" not in result.sanitized_text
        assert "[X:" in result.sanitized_text
        assert result.sanitized_text.startswith("prefix ")
        assert result.sanitized_text.endswith(" suffix")

    def test_two_replacements_order_preserved(self):
        text  = "A: FIRST | B: SECOND"
        spans = [
            DetectedSpan("TYPE1", "FIRST",  3,  8),
            DetectedSpan("TYPE2", "SECOND", 14, 20),
        ]
        result = _engine(spans, {"TYPE1": "hash", "TYPE2": "hash"}).scan(text)
        assert "FIRST"  not in result.sanitized_text
        assert "SECOND" not in result.sanitized_text
        assert "[TYPE1:" in result.sanitized_text
        assert "[TYPE2:" in result.sanitized_text
        # TYPE1 placeholder TYPE2 placeholder'dan önce gelmeli
        assert result.sanitized_text.index("[TYPE1:") < result.sanitized_text.index("[TYPE2:")

    def test_five_replacements_all_correct(self):
        """5 ardışık entity'nin hepsi doğru konumda replace edilmeli."""
        # Entity type adlarından farklı değerler kullan; placeholder içinde tip adı geçer
        text  = "a:CARD b:TCNO c:MAIL d:PHONE e:IBAN_NUM"
        # Pozisyonları metinden hesapla — manuel hata riskini ortadan kaldırır
        def _pos(word): s = text.index(word); return s, s + len(word)
        spans = [
            DetectedSpan("CC", "CARD",     *_pos("CARD")),
            DetectedSpan("TC", "TCNO",     *_pos("TCNO")),
            DetectedSpan("EM", "MAIL",     *_pos("MAIL")),
            DetectedSpan("PH", "PHONE",    *_pos("PHONE")),
            DetectedSpan("IB", "IBAN_NUM", *_pos("IBAN_NUM")),
        ]
        actions = {t: "hash" for t in ["CC", "TC", "EM", "PH", "IB"]}
        result = _engine(spans, actions).scan(text)

        # Orijinal içerik değerleri sanitized metinde olmamalı
        for original in ["CARD", "TCNO", "MAIL", "PHONE", "IBAN_NUM"]:
            assert original not in result.sanitized_text, \
                f"'{original}' sanitized metinde hâlâ var: {result.sanitized_text!r}"

        # Tüm placeholder'lar metin içinde bulunmalı
        for t in ["CC", "TC", "EM", "PH", "IB"]:
            assert f"[{t}:" in result.sanitized_text

        # Sabit metin bölümleri (a:, b:, vb.) korunmalı
        for label in ["a:", "b:", "c:", "d:", "e:"]:
            assert label in result.sanitized_text

    def test_warn_does_not_shift_offset(self):
        """warn action offset'i değiştirmemeli; sonraki hash doğru konumda olmalı."""
        text  = "W: WARN | H: HASH"
        spans = [
            DetectedSpan("W", "WARN", 3,  7),
            DetectedSpan("H", "HASH", 13, 17),
        ]
        result = _engine(spans, {"W": "warn", "H": "hash"}).scan(text)
        assert "WARN" in result.sanitized_text   # warn → değişmez
        assert "HASH" not in result.sanitized_text
        assert "[H:" in result.sanitized_text

    def test_replacement_longer_than_original(self):
        """Replacement orijinalden uzunsa sonraki span yine doğru yerde olmalı."""
        text  = "A: X | B: Y"
        # 'X' → '[VERY_LONG_TYPE_NAME:abcd1234]' (çok daha uzun)
        spans = [
            DetectedSpan("VERY_LONG_TYPE_NAME", "X", 3, 4),
            DetectedSpan("B",                   "Y", 10, 11),
        ]
        result = _engine(spans, {"VERY_LONG_TYPE_NAME": "hash", "B": "hash"}).scan(text)
        assert "[VERY_LONG_TYPE_NAME:" in result.sanitized_text
        assert "[B:" in result.sanitized_text

    def test_replacement_shorter_than_original(self):
        """Replacement orijinalden kısa olsa bile sonraki span doğru yerde olmalı."""
        text  = "AAAAAAAAAA | B: Y"
        spans = [
            DetectedSpan("LONG", "AAAAAAAAAA", 0, 10),  # 10 char → ~22 char placeholder
            DetectedSpan("B",    "Y",           15, 16),
        ]
        result = _engine(spans, {"LONG": "hash", "B": "hash"}).scan(text)
        assert "[LONG:" in result.sanitized_text
        assert "[B:" in result.sanitized_text


# ═══════════════════════════════════════════════════════════════════════════
# Violation sıralama
# ═══════════════════════════════════════════════════════════════════════════

class TestViolationOrdering:
    def test_violations_in_text_order(self):
        """Violations, metindeki görünüm sırasında dönmeli."""
        text  = "first: ALPHA second: BETA third: GAMMA"
        spans = [
            DetectedSpan("C", "GAMMA", 33, 38),
            DetectedSpan("A", "ALPHA", 7,  12),
            DetectedSpan("B", "BETA",  21, 25),
        ]
        result = _engine(spans, {"A": "warn", "B": "warn", "C": "warn"}).scan(text)
        entity_order = [v.entity_type for v in result.violations]
        assert entity_order == ["A", "B", "C"]

    def test_violation_start_positions_ascending(self):
        from ai_guard import LLMGuard
        guard = LLMGuard(use_ner=False)
        text  = "a@a.com 0532 111 22 33 b@b.com"
        result = guard.scan(text)
        starts = [v.start for v in result.violations]
        assert starts == sorted(starts)


# ═══════════════════════════════════════════════════════════════════════════
# Boş / minimal girdi
# ═══════════════════════════════════════════════════════════════════════════

class TestEmptyAndMinimalInput:
    def test_empty_string(self):
        result = _engine([], {}).scan("")
        assert result.is_clean
        assert result.sanitized_text == ""
        assert result.original_text  == ""

    def test_whitespace_only(self):
        result = _engine([], {}).scan("   \n\t  ")
        assert result.is_clean

    def test_single_char(self):
        result = _engine([], {}).scan("x")
        assert result.is_clean

    def test_no_detectors_means_clean(self):
        engine = DetectionEngine({"salt": "", "entities": {}}, [])
        result = engine.scan("kart: 4111111111111111 TC: 12345678901")
        assert result.is_clean

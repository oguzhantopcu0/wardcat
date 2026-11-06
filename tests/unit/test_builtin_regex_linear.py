"""Every built-in pattern must scan adversarial input in linear time.

``test_builtin_redos`` catches catastrophic (exponential) backtracking on short
inputs. This catches the quieter kind: a pattern that restarts at every
character of a long run and rescans the rest of it, which is quadratic. ADDRESS
on ``"A" * n`` and FINANCIAL_AMOUNT on ``"1." * n`` were like that — 20 000
characters took seconds, the 500 KB input limit about half an hour.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from wardcat.detectors.regex_detector import _COMPILED

N = 50_000

# Runs a regex can be made to restart in: letters, capitalized words, digits
# with separators, the characters around an "@".
ADVERSARIAL: dict[str, Callable[[int], str]] = {
    "A*n": lambda n: "A" * n,
    "a*n": lambda n: "a" * n,
    "Aa*n": lambda n: "Aa" * (n // 2),
    "A_*n": lambda n: "A " * (n // 2),
    "Ab_*n": lambda n: "Ab " * (n // 3),
    "1*n": lambda n: "1" * n,
    "1.*n": lambda n: "1." * (n // 2),
    "1,*n": lambda n: "1," * (n // 2),
    "1_*n": lambda n: "1 " * (n // 2),
    "a@*n": lambda n: "a@" * (n // 2),
    "a.*n": lambda n: "a." * (n // 2),
    "lira-1.*n": lambda n: "₺" + "1." * (n // 2),
}


def _seconds(pattern, text: str) -> float:
    best = float("inf")
    for _ in range(2):
        start = time.perf_counter()
        for _match in pattern.finditer(text):
            pass
        best = min(best, time.perf_counter() - start)
    return best


@pytest.mark.parametrize("entity_type", sorted(_COMPILED))
def test_a_builtin_pattern_is_linear_on_adversarial_input(entity_type: str) -> None:
    pattern = _COMPILED[entity_type]
    for name, make in ADVERSARIAL.items():
        doubled = _seconds(pattern, make(2 * N))
        # Too quick to be quadratic at this size; a ratio of two tiny numbers is
        # timer noise, not growth.
        if doubled < 0.05:
            continue
        single = _seconds(pattern, make(N))
        assert doubled < 0.5, f"{entity_type} on {name}: {doubled:.2f}s for {2 * N} chars"
        assert doubled / single < 2.5, (
            f"{entity_type} on {name}: {single:.3f}s → {doubled:.3f}s when the input doubles"
        )


@pytest.mark.parametrize(
    ("entity_type", "text", "expected"),
    [
        # The anchors must not cost real matches.
        ("ADDRESS", "Adres: Atatürk Caddesi No: 5", "Atatürk Caddesi No: 5"),
        ("ADDRESS", "(Bağdat Caddesi No: 12)", "Bağdat Caddesi No: 12"),
        ("ADDRESS", "İnönü Mahallesi Gül Sokak No 3", "İnönü Mahallesi Gül Sokak No 3"),
        ("FINANCIAL_AMOUNT", "toplam 85.000 TL ödendi", "85.000 TL"),
        ("FINANCIAL_AMOUNT", "kalem,500 TL", "500 TL"),
        ("FINANCIAL_AMOUNT", "bedel:47,3 milyon TL", "47,3 milyon TL"),
        ("FINANCIAL_AMOUNT", "₺47.3 milyon", "₺47.3 milyon"),
    ],
)
def test_the_anchored_patterns_still_match(entity_type: str, text: str, expected: str) -> None:
    found = [m.group() for m in _COMPILED[entity_type].finditer(text)]
    assert expected in found, found


def test_a_number_is_not_matched_from_its_middle() -> None:
    found = [m.group() for m in _COMPILED["FINANCIAL_AMOUNT"].finditer("1.250.000 TL")]
    assert found == ["1.250.000 TL"]

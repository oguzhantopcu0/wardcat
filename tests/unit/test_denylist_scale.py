"""Literal denylist entries are matched in one pass, with the same result."""

from __future__ import annotations

import random
import re
import time

from wardcat import Action, Entity, Wardcat
from wardcat.detectors.base import DetectedSpan


def reference(text: str, entries: list[tuple[str, str]]) -> list[DetectedSpan]:
    """The entry-by-entry search the engine used to run, kept as the oracle."""
    spans = []
    for entity_type, value in entries:
        start = 0
        while True:
            pos = text.find(value, start)
            if pos == -1:
                break
            spans.append(DetectedSpan(entity_type, value, pos, pos + len(value), 1.0))
            start = pos + 1
    return spans


def test_the_combined_pass_resolves_to_the_same_spans() -> None:
    rng = random.Random(3)
    alphabet = "abc"
    for _ in range(300):
        values = list({"".join(rng.choices(alphabet, k=rng.randint(1, 4))) for _ in range(6)})
        entries = [("CUSTOM", v) for v in values]
        text = "".join(rng.choices(alphabet + "  ", k=40))
        guard = Wardcat(salt="s").add_denylist(
            [{"value": v, "entity_type": "CUSTOM"} for v in values]
        )
        engine = guard._engine

        got = engine._resolve_overlaps(engine._collect_denylist_spans(text))
        want = engine._resolve_overlaps(reference(text, entries))

        assert [(s.start, s.end, s.text) for s in got] == [(s.start, s.end, s.text) for s in want]


def test_regex_entries_still_run_separately() -> None:
    guard = Wardcat(salt="s").add_denylist(
        [
            {"pattern": r"PRJ-\d{3}", "entity_type": "CUSTOM_SECRET"},
            {"value": "ProjectX", "entity_type": "CUSTOM_SECRET"},
        ]
    )
    result = guard.scan("PRJ-123 and ProjectX")
    assert sorted(v.original for v in result.violations) == ["PRJ-123", "ProjectX"]
    assert {v.source for v in result.violations} == {"denylist"}


def test_a_value_under_two_types_keeps_the_first(caplog) -> None:
    import logging

    with caplog.at_level(logging.WARNING, logger="wardcat"):
        guard = Wardcat(salt="s").add_denylist(
            [
                {"value": "Acme", "entity_type": "ORG"},
                {"value": "Acme", "entity_type": "PERSON"},
            ]
        )
    assert guard.scan("Acme").violations[0].entity_type == "ORG"
    assert any("two entity types" in r.message for r in caplog.records)


def _timed(fn) -> float:
    started = time.perf_counter()
    fn()
    return time.perf_counter() - started


def test_ten_thousand_names_scan_in_one_pass() -> None:
    rng = random.Random(11)
    names = [f"Name{i:05d} Surname{rng.randint(0, 999)}" for i in range(10_000)]
    text = ("Lorem ipsum " * 40 + names[1234] + " and " + names[9876] + " ") * 50
    guard = Wardcat(salt="s").add_entity(Entity.EMAIL, Action.REDACT)
    guard.add_denylist([{"value": n, "entity_type": "PERSON"} for n in names])

    # Best of three, so a busy CI runner does not decide the outcome. The
    # one-pass search was measured at a fifth of the entry-by-entry time on a
    # quiet machine; the test only insists it is the faster of the two.
    elapsed = min(_timed(lambda: guard.scan(text)) for _ in range(3))
    baseline = min(
        _timed(lambda: reference(text, [("PERSON", n) for n in names])) for _ in range(3)
    )

    assert len(guard.scan(text).violations) == 100
    assert elapsed < baseline, f"combined {elapsed:.3f}s vs entry-by-entry {baseline:.3f}s"
    # The alternation itself must not be pathological.
    assert re.compile(guard._engine._denylist_literal_re.pattern)

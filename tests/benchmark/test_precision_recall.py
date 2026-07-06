"""Precision/recall regression gate.

Runs the labelled-corpus harness and asserts the regex detector still catches
every known PII value (recall) without over-flagging (precision). The corpus is
curated to be fully regex-detectable, so the bar is exact: any drop means a
detector regressed. Add rows to ``eval_harness.CORPUS`` to widen coverage.
"""

from __future__ import annotations

from tests.benchmark.eval_harness import evaluate


def test_micro_average_is_perfect_on_curated_corpus():
    report = evaluate()
    micro = report.micro
    assert micro.recall == 1.0, f"recall regressed: {report.format_table()}"
    assert micro.precision == 1.0, (
        f"precision regressed (false positives):\n{report.format_table()}"
    )


def test_every_entity_has_full_recall():
    report = evaluate()
    missed = {e: s.recall for e, s in report.per_entity.items() if s.recall < 1.0}
    assert not missed, f"entities with missed detections: {missed}\n{report.format_table()}"


def test_no_false_positives_per_entity():
    report = evaluate()
    noisy = {e: s.fp for e, s in report.per_entity.items() if s.fp > 0}
    assert not noisy, f"entities producing false positives: {noisy}\n{report.format_table()}"

"""scan_batch: batched layers in one pass, the rest bounded, results unchanged."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from unittest.mock import MagicMock

import pytest

from wardcat import Action, DegradedScanError, Entity, Wardcat
from wardcat.detectors.base import BaseDetector, DetectedSpan
from wardcat.llm.backends.base import BaseLLMBackend

TEXTS = [
    "mail ali@example.com and card 4111 1111 1111 1111",
    "nothing here",
    "Phone: 905-674-3793, ProjectX",
    "",
    "iban TR33 0006 1005 1978 6457 8413 26",
] * 4


class BatchingDetector(BaseDetector):
    """Finds the word 'ProjectX'; counts how it was called."""

    layer = "custom"

    def __init__(self) -> None:
        self.single_calls = 0
        self.batch_calls = 0

    def _find(self, text: str) -> list[DetectedSpan]:
        i = text.find("ProjectX")
        return [] if i < 0 else [DetectedSpan("CUSTOM_SECRET", "ProjectX", i, i + 8, 0.97)]

    def detect(self, text, candidates=None):
        self.single_calls += 1
        return self._find(text)

    def detect_many(self, texts):
        self.batch_calls += 1
        return [self._find(t) for t in texts]


def strip_ids(result) -> tuple:
    return (
        result.sanitized_text,
        [
            (v.entity_type, v.original, v.start, v.end, v.action, v.source)
            for v in result.violations
        ],
        tuple(result.warnings),
        result.scan_error,
    )


class TestEquivalence:
    def test_batch_equals_one_scan_per_text(self) -> None:
        guard = Wardcat(salt="s").add_entities(
            [Entity.EMAIL, Entity.CREDIT_CARD, Entity.PHONE, Entity.IBAN, Entity.CUSTOM_SECRET],
            action=Action.REDACT,
        )
        detector = BatchingDetector()
        guard._engine.detectors.append(detector)

        batched = guard.scan_batch(TEXTS)
        single = [guard.scan(t) for t in TEXTS]

        assert [strip_ids(r) for r in batched] == [strip_ids(r) for r in single]
        assert detector.batch_calls == 1
        assert detector.single_calls == len(TEXTS)  # from the per-text scans only

    def test_context_ids_are_distinct_per_item(self) -> None:
        guard = Wardcat(salt="s").add_entity(Entity.EMAIL, Action.TOKENIZE)
        results = guard.scan_batch(["a@b.io", "c@d.io"])
        assert results[0].context_id != results[1].context_id

    def test_an_oversized_item_gets_a_scan_error_and_the_rest_are_scanned(self) -> None:
        guard = Wardcat(salt="s").add_entity(Entity.EMAIL, Action.REDACT)
        guard._config["max_text_bytes"] = 100
        guard._rebuild()
        results = guard.scan_batch(["a@b.io", "x" * 200, "c@d.io"])
        assert results[0].sanitized_text == "[EMAIL]"
        assert results[1].scan_error and results[1].sanitized_text == "x" * 200
        assert results[2].sanitized_text == "[EMAIL]"

    def test_a_batched_layer_that_fails_is_reported_on_every_item(self) -> None:
        class Broken(BaseDetector):
            def detect(self, text, candidates=None):
                return []

            def detect_many(self, texts):
                raise RuntimeError("model crashed")

        guard = Wardcat(salt="s").add_entity(Entity.EMAIL, Action.REDACT)
        guard._engine.detectors.append(Broken())
        results = guard.scan_batch(["a@b.io", "c@d.io"])
        assert all(any("model crashed" in w for w in r.warnings) for r in results)
        assert results[0].sanitized_text == "[EMAIL]"

    def test_strict_still_raises_from_a_batch(self) -> None:
        class Broken(BaseDetector):
            def detect(self, text, candidates=None):
                raise RuntimeError("down")

        guard = Wardcat(salt="s").add_entity(Entity.EMAIL, Action.REDACT).with_strict()
        guard._engine.detectors.append(Broken())
        with pytest.raises(DegradedScanError):
            guard.scan_batch(["a@b.io"])


class TestConcurrencyGate:
    def _guard(self, max_concurrency: int, delay: float = 0.05):
        peak = {"now": 0, "max": 0}
        lock = threading.Lock()

        def reply(*_a, **_k):
            with lock:
                peak["now"] += 1
                peak["max"] = max(peak["max"], peak["now"])
            time.sleep(delay)
            with lock:
                peak["now"] -= 1
            return json.dumps({"entities": []})

        backend = MagicMock(spec=BaseLLMBackend)
        backend.complete_messages.side_effect = reply

        async def reply_async(*a, **k):
            return await asyncio.to_thread(reply)

        backend.complete_messages_async.side_effect = reply_async
        guard = (
            Wardcat(salt="s")
            .with_llm(model="x", max_concurrency=max_concurrency)
            .add_entity(Entity.PERSON, Action.REDACT, layers=["llm"])
        )
        guard._engine.detectors[-1].backend = backend
        return guard, peak

    def test_sync_batch_never_exceeds_max_concurrency(self) -> None:
        guard, peak = self._guard(2)
        guard.scan_batch([f"text {i}" for i in range(12)], max_workers=8)
        assert 1 <= peak["max"] <= 2

    def test_async_batch_never_exceeds_max_concurrency(self) -> None:
        guard, peak = self._guard(3)
        asyncio.run(guard.scan_batch_async([f"text {i}" for i in range(12)], max_workers=8))
        assert 1 <= peak["max"] <= 3

    def test_yaml_validation(self, tmp_path) -> None:
        from wardcat import ConfigError

        cfg = tmp_path / "p.yaml"
        cfg.write_text("llm_detector:\n  enabled: false\n  max_concurrency: 0\n")
        with pytest.raises(ConfigError):
            Wardcat(config_path=str(cfg))


@pytest.mark.ner
def test_the_pipe_path_matches_the_single_path() -> None:
    from wardcat.detectors.ner_detector import NERDetector

    detector = NERDetector({"PERSON", "ORG", "LOCATION"}, model="en_core_web_sm")
    texts = [
        "John Smith met Jane Doe at Acme Corp in London.",
        "Nothing to see.",
        "Dr. Elena Petrova and Microsoft signed in Berlin.",
    ]
    batched = detector.detect_many(texts)
    single = [detector.detect(t) for t in texts]
    assert batched == single

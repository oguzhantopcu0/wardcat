"""An LLM that fails must never leave the scan with fewer detections than no LLM.

Under adjudication, a candidate below the keep threshold (an NER name, 0.85)
survives only when the model confirms it. Until 1.2.2 a model that timed out,
answered with something other than a JSON list, or could not be reached at all
"confirmed" nothing, so the name went out unmasked: silently in the first two
cases, with only a warning in the third. Each case now keeps the candidates the
model never judged and says so on the result, and strict mode refuses the scan.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from wardcat.core.engine import DetectionEngine
from wardcat.detectors.base import BaseDetector, DetectedSpan
from wardcat.detectors.llm_detector import LLMDetector, UnreadableReply
from wardcat.exceptions import DegradedScanError
from wardcat.llm.backends.base import BaseLLMBackend
from wardcat.llm.circuit import CircuitBreaker

TEXT = "Ali Veli called today."
NAME = DetectedSpan("PERSON", "Ali Veli", 0, 8, 0.85)


class FixedDetector(BaseDetector):
    layer = "ner"

    def __init__(self, spans: list[DetectedSpan]) -> None:
        self._spans = spans

    def detect(self, text, candidates=None):
        return [
            DetectedSpan(s.entity_type, s.text, s.start, s.end, s.confidence) for s in self._spans
        ]


def backend(reply: str | None = None, error: Exception | None = None) -> MagicMock:
    mock = MagicMock(spec=BaseLLMBackend)
    mock.complete_messages_async = AsyncMock()
    if error is not None:
        mock.complete_messages.side_effect = error
        mock.complete_messages_async.side_effect = error
    else:
        mock.complete_messages.return_value = reply
        mock.complete_messages_async.return_value = reply
    return mock


def engine(llm: LLMDetector, *, adjudicate: bool = True, strict: bool = False, spans=None):
    config = {
        "salt": "s",
        "strict": strict,
        "entities": {"PERSON": {"enabled": True, "action": "redact"}},
        "llm_detector": {"adjudicate": adjudicate},
    }
    return DetectionEngine(config, [FixedDetector(spans or [NAME]), llm])


def both(eng: DetectionEngine, text: str = TEXT):
    return eng.scan(text), asyncio.run(eng.scan_async(text))


FAILURES = {
    "unreadable reply": {"reply": "Sure! I could not find anything worth flagging."},
    "broken json": {"reply": '[{"type": "PERSON", "text": "Ali Veli"'},
    "json object": {"reply": '{"type": "PERSON"}'},
    "chunk timeout": {"error": TimeoutError("timed out")},
    "backend down": {"error": ConnectionError("connection refused")},
}


@pytest.mark.parametrize("case", sorted(FAILURES))
class TestAFailingModelKeepsTheCandidates:
    def test_the_name_is_masked(self, case: str) -> None:
        llm = LLMDetector(backend(**FAILURES[case]), {"PERSON"})
        for result in both(engine(llm)):
            assert result.sanitized_text == "[PERSON] called today."

    def test_the_failure_is_on_the_result(self, case: str) -> None:
        llm = LLMDetector(backend(**FAILURES[case]), {"PERSON"})
        for result in both(engine(llm)):
            assert len(result.warnings) == 1
            assert "kept 1 candidate span(s)" in result.warnings[0]

    def test_strict_refuses_the_scan(self, case: str) -> None:
        llm = LLMDetector(backend(**FAILURES[case]), {"PERSON"})
        eng = engine(llm, strict=True)
        with pytest.raises(DegradedScanError):
            eng.scan(TEXT)
        with pytest.raises(DegradedScanError):
            asyncio.run(eng.scan_async(TEXT))

    def test_no_value_reaches_the_warning(self, case: str) -> None:
        llm = LLMDetector(backend(**FAILURES[case]), {"PERSON"})
        for result in both(engine(llm)):
            assert "Ali" not in " ".join(result.warnings)


class TestAnOpenCircuit:
    def test_counts_as_the_backend_being_down(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1, cooldown=60)
        breaker.record_failure()  # open
        llm = LLMDetector(backend(reply="[]"), {"PERSON"}, breaker=breaker)
        for result in both(engine(llm)):
            assert result.sanitized_text == "[PERSON] called today."
            assert "kept 1 candidate span(s)" in result.warnings[0]


class TestRealAnswersAreUnchanged:
    def test_a_confirmation_masks_without_warning(self) -> None:
        reply = '[{"type": "PERSON", "text": "Ali Veli"}]'
        llm = LLMDetector(backend(reply=reply), {"PERSON"})
        for result in both(engine(llm)):
            assert result.sanitized_text == "[PERSON] called today."
            assert result.warnings == []

    def test_an_empty_list_is_an_answer(self) -> None:
        # "Nothing here" is a verdict, not a failure: the candidate is dropped
        # and nothing is reported. (Limiting how much the model may drop is the
        # next release's adjudication budget.)
        llm = LLMDetector(backend(reply="[]"), {"PERSON"})
        for result in both(engine(llm)):
            assert result.sanitized_text == TEXT
            assert result.warnings == []

    def test_regex_strength_spans_need_no_rescue(self) -> None:
        card = DetectedSpan("PERSON", "Ali Veli", 0, 8, 1.0)
        llm = LLMDetector(backend(error=ConnectionError("down")), {"PERSON"})
        result = engine(llm, spans=[card]).scan(TEXT)
        assert result.sanitized_text == "[PERSON] called today."
        assert "kept 0 candidate span(s)" in result.warnings[0]


class TestWithoutAdjudication:
    def test_an_unreadable_chunk_is_reported(self) -> None:
        llm = LLMDetector(backend(reply="no list here"), {"PERSON"})
        for result in both(engine(llm, adjudicate=False)):
            assert result.sanitized_text == "[PERSON] called today."  # NER ran on its own
            assert result.warnings == [
                "LLMDetector: the chunk at offset 0 was not judged (UnreadableReply)"
            ]


class TestOnlyTheFailedChunkFallsBack:
    def test_a_good_chunk_still_decides_its_own_candidates(self) -> None:
        text = "Ali Veli called.\n\nZeynep Ak wrote back."
        spans = [
            DetectedSpan("PERSON", "Ali Veli", 0, 8, 0.85),
            DetectedSpan("PERSON", "Zeynep Ak", 18, 27, 0.85),
        ]
        mock = MagicMock(spec=BaseLLMBackend)
        # First chunk: the model clears "Ali Veli". Second chunk: it times out.
        mock.complete_messages.side_effect = ["[]", TimeoutError("slow")]
        llm = LLMDetector(mock, {"PERSON"}, chunk_chars=20)
        result = engine(llm, spans=spans).scan(text)
        assert result.sanitized_text == "Ali Veli called.\n\n[PERSON] wrote back."
        assert result.warnings == [
            "LLMDetector: the chunk at offset 18 was not judged (TimeoutError); "
            "kept 1 candidate span(s) from the other layers"
        ]

    def test_a_candidate_no_chunk_contains_is_kept(self) -> None:
        # An address block runs over a line break, and the chunker splits there:
        # no chunk holds the whole span, so the model never sees it. It is kept.
        text = "Atatürk Cad.\nNo 5 Kadıköy\n\nThanks."
        address = DetectedSpan("ADDRESS", "Atatürk Cad.\nNo 5 Kadıköy", 0, 25, 0.85)
        llm = LLMDetector(backend(reply="[]"), {"ADDRESS"}, chunk_chars=14)
        assert len(llm._to_chunks(text)) > 1
        report = llm.detect_report(text, [address])
        assert [s.text for s in report.spans] == [address.text]
        assert report.warnings == []


class TestFailuresAreNotCached:
    def test_the_next_scan_asks_the_model_again(self) -> None:
        mock = MagicMock(spec=BaseLLMBackend)
        mock.complete_messages.side_effect = [
            TimeoutError("slow"),
            '[{"type": "PERSON", "text": "Ali Veli"}]',
        ]
        llm = LLMDetector(mock, {"PERSON"}, cache_ttl=300)
        eng = engine(llm)
        first = eng.scan(TEXT)
        second = eng.scan(TEXT)
        assert first.warnings and not second.warnings
        assert mock.complete_messages.call_count == 2


def test_unreadable_reply_is_a_value_error() -> None:
    # The chunk loop catches ValueError; an UnreadableReply must stay one.
    assert issubclass(UnreadableReply, ValueError)

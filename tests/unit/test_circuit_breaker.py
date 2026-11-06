"""The LLM layer stops calling a backend that keeps failing."""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest

from wardcat import Action, CircuitOpen, ConfigError, Entity, Wardcat
from wardcat.detectors.llm_detector import LLMDetector
from wardcat.llm.backends.base import BaseLLMBackend
from wardcat.llm.circuit import CircuitBreaker


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TestBreaker:
    def test_opens_after_the_threshold_and_reports_the_wait(self) -> None:
        clock = FakeClock()
        breaker = CircuitBreaker(3, 30, clock=clock)
        for _ in range(3):
            breaker.check()
            breaker.record_failure()

        with pytest.raises(CircuitOpen) as info:
            breaker.check()

        assert info.value.failures == 3
        assert info.value.retry_after == pytest.approx(30)
        assert "3 consecutive failures" in str(info.value)

    def test_a_success_before_the_threshold_resets_the_count(self) -> None:
        breaker = CircuitBreaker(3, 30, clock=FakeClock())
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        breaker.record_failure()

        breaker.check()  # still closed: only two in a row

    def test_one_trial_after_the_cooldown_then_closed_on_success(self) -> None:
        clock = FakeClock()
        breaker = CircuitBreaker(2, 10, clock=clock)
        breaker.record_failure()
        breaker.record_failure()
        clock.now += 10

        breaker.check()  # the trial
        with pytest.raises(CircuitOpen):
            breaker.check()  # only one trial at a time
        breaker.record_success()

        breaker.check()
        assert not breaker.is_open

    def test_a_failed_trial_reopens_for_another_cooldown(self) -> None:
        clock = FakeClock()
        breaker = CircuitBreaker(2, 10, clock=clock)
        breaker.record_failure()
        breaker.record_failure()
        clock.now += 10
        breaker.check()
        breaker.record_failure()

        with pytest.raises(CircuitOpen) as info:
            breaker.check()
        assert info.value.retry_after == pytest.approx(10)

    def test_zero_threshold_disables_it(self) -> None:
        breaker = CircuitBreaker(0, 30)
        for _ in range(10):
            breaker.record_failure()
        breaker.check()
        assert not breaker.enabled

    def test_counts_are_consistent_under_threads(self) -> None:
        breaker = CircuitBreaker(500, 30)

        def hammer() -> None:
            for _ in range(100):
                breaker.record_failure()

        threads = [threading.Thread(target=hammer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert breaker._failures == 800
        assert breaker.is_open


def _backend(side_effect) -> BaseLLMBackend:
    backend = MagicMock(spec=BaseLLMBackend)
    backend.complete_messages.side_effect = side_effect
    return backend


class TestDetector:
    def test_the_backend_is_not_called_while_open(self) -> None:
        clock = FakeClock()
        backend = _backend(ConnectionError("down"))
        detector = LLMDetector(
            backend=backend,
            enabled_entities={"EMAIL"},
            breaker=CircuitBreaker(2, 30, clock=clock),
        )
        for _ in range(2):
            with pytest.raises(ConnectionError):
                detector.detect("x ali@example.com")
        with pytest.raises(CircuitOpen):
            detector.detect("x ali@example.com")

        assert backend.complete_messages.call_count == 2

    def test_http_timeouts_count_as_failures(self) -> None:
        backend = _backend(TimeoutError("slow"))
        detector = LLMDetector(
            backend=backend, enabled_entities={"EMAIL"}, breaker=CircuitBreaker(1, 30)
        )
        detector.detect("x")  # timeout is swallowed per chunk, but counted
        with pytest.raises(CircuitOpen):
            detector.detect("x")

    def test_an_unparseable_reply_is_not_a_failure(self) -> None:
        backend = _backend("not json")
        detector = LLMDetector(
            backend=backend, enabled_entities={"EMAIL"}, breaker=CircuitBreaker(1, 30)
        )
        detector.detect("x")
        detector.detect("x")
        assert backend.complete_messages.call_count == 2

    def test_async_path_shares_the_breaker(self) -> None:
        backend = MagicMock(spec=BaseLLMBackend)

        async def _down(*_a, **_k):
            raise ConnectionError("down")

        backend.complete_messages_async.side_effect = _down
        detector = LLMDetector(
            backend=backend, enabled_entities={"EMAIL"}, breaker=CircuitBreaker(1, 30)
        )
        with pytest.raises(ConnectionError):
            asyncio.run(detector.detect_async("x"))
        with pytest.raises(CircuitOpen):
            asyncio.run(detector.detect_async("x"))


class TestAnAbandonedTrial:
    """A half-open trial that ends without a verdict must give the trial back.

    A cancelled call (``asyncio.CancelledError``) or an interrupt is a
    ``BaseException``: it recorded neither success nor failure, so the trial
    flag stayed set and every later call got ``CircuitOpen`` — the LLM layer
    was off until the process restarted.
    """

    @staticmethod
    def _open(clock: FakeClock) -> CircuitBreaker:
        breaker = CircuitBreaker(1, 10, clock=clock)
        breaker.record_failure()
        clock.now += 10
        return breaker

    def test_a_cancelled_async_trial_lets_the_next_call_try(self) -> None:
        clock = FakeClock()
        breaker = self._open(clock)
        calls = []

        async def _reply(*_a, **_k):
            calls.append(1)
            if len(calls) == 1:
                await asyncio.sleep(10)
            return "[]"

        backend = MagicMock(spec=BaseLLMBackend)
        backend.complete_messages_async.side_effect = _reply
        detector = LLMDetector(backend=backend, enabled_entities={"EMAIL"}, breaker=breaker)

        async def _run() -> str:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(detector.complete_messages_async([]), 0.01)
            return await detector.complete_messages_async([])

        assert asyncio.run(_run()) == "[]"
        assert len(calls) == 2
        assert not breaker.is_open

    def test_an_interrupted_sync_trial_lets_the_next_call_try(self) -> None:
        clock = FakeClock()
        breaker = self._open(clock)
        backend = MagicMock(spec=BaseLLMBackend)
        backend.complete_messages.side_effect = [KeyboardInterrupt(), "[]"]
        detector = LLMDetector(backend=backend, enabled_entities={"EMAIL"}, breaker=breaker)

        with pytest.raises(KeyboardInterrupt):
            detector.complete_messages([])
        assert detector.complete_messages([]) == "[]"
        assert not breaker.is_open

    def test_an_abandoned_trial_is_not_a_failure(self) -> None:
        clock = FakeClock()
        breaker = self._open(clock)
        assert breaker.check() is True  # this caller holds the trial
        breaker.release_trial()
        # Still open, the cooldown not restarted: the next caller tries at once.
        assert breaker.is_open
        assert breaker.check() is True

    def test_releasing_does_not_free_someone_elses_trial(self) -> None:
        clock = FakeClock()
        breaker = CircuitBreaker(1, 10, clock=clock)
        assert breaker.check() is False  # closed: no trial is held
        breaker.record_failure()
        clock.now += 10
        assert breaker.check() is True
        backend = MagicMock(spec=BaseLLMBackend)
        backend.complete_messages.side_effect = KeyboardInterrupt()
        # A caller that passed check() while the circuit was closed and is
        # interrupted now must not release the trial another caller holds.
        detector = LLMDetector(backend=backend, enabled_entities={"EMAIL"}, breaker=breaker)
        breaker._opened_at = None  # let this caller through as if closed
        with pytest.raises(KeyboardInterrupt):
            detector.complete_messages([])
        breaker._opened_at = clock.now - 10
        with pytest.raises(CircuitOpen):
            breaker.check()


class TestThroughTheGuard:
    def _guard(self, side_effect, **llm) -> Wardcat:
        guard = (
            Wardcat(salt="s")
            .with_llm(model="x", timeout=1, **llm)
            .add_entity(Entity.EMAIL, Action.REDACT)
        )
        detector = guard._engine.detectors[-1]
        detector.backend = _backend(side_effect)
        return guard

    def test_scans_under_an_open_circuit_are_degraded_and_say_so(self) -> None:
        guard = self._guard(ConnectionError("down"), circuit_failures=2, circuit_cooldown=30)
        guard.scan("a")
        guard.scan("b")

        result = guard.scan("c ali@example.com")

        assert result.sanitized_text == "c [EMAIL]"
        assert any("circuit open" in w for w in result.warnings)
        assert guard._engine.detectors[-1].backend.complete_messages.call_count == 2

    def test_ten_scans_against_a_dead_backend_finish_fast(self) -> None:
        def slow_failure(*_a, **_k):
            time.sleep(1)
            raise ConnectionError("down")

        guard = self._guard(slow_failure, circuit_failures=3, circuit_cooldown=60)
        started = time.perf_counter()
        for _ in range(10):
            guard.scan("x")
        assert time.perf_counter() - started < 3.5

    def test_is_sensitive_raises_while_open(self) -> None:
        guard = self._guard(ConnectionError("down"), circuit_failures=1, circuit_cooldown=30)
        with pytest.raises(ConnectionError):
            guard.is_sensitive("secret")
        with pytest.raises(CircuitOpen):
            guard.is_sensitive("secret")

    def test_zero_disables_it(self) -> None:
        guard = self._guard(ConnectionError("down"), circuit_failures=0)
        for _ in range(5):
            guard.scan("x")
        assert guard._engine.detectors[-1].backend.complete_messages.call_count == 5

    def test_yaml_validation(self, tmp_path) -> None:
        cfg = tmp_path / "p.yaml"
        cfg.write_text("llm_detector:\n  enabled: false\n  circuit_failures: -1\n")
        with pytest.raises(ConfigError):
            Wardcat(config_path=str(cfg))

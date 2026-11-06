"""Fail fast while an LLM backend is down.

A backend that is unreachable makes every scan wait the full request timeout
before the layer is skipped — 60 seconds by default, on every call, for as long
as the outage lasts. The breaker counts consecutive backend failures; once they
reach the threshold it refuses calls for a cooldown, then lets one trial through.
A refused call raises :class:`CircuitOpen`, which the engine records on the
result as a warning like any other skipped layer, so a scan under an open
circuit is *degraded*, never silently regex-only.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from wardcat.exceptions import WardcatError


class CircuitOpen(WardcatError):
    """The LLM backend is being skipped after repeated failures.

    ``retry_after`` is how many seconds remain before the next trial call.
    """

    def __init__(self, failures: int, retry_after: float) -> None:
        self.failures = failures
        self.retry_after = retry_after
        super().__init__(
            f"LLM layer skipped: backend circuit open for another {retry_after:.0f} s "
            f"after {failures} consecutive failures"
        )


class CircuitBreaker:
    """Consecutive-failure breaker with a single half-open trial.

    :param failure_threshold: failures in a row that open the circuit; ``0``
        disables the breaker entirely.
    :param cooldown: seconds the circuit stays open before one trial call is let
        through. A failed trial reopens it for another cooldown.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown: float = 30.0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 0:
            raise ValueError("failure_threshold must be >= 0")
        if cooldown < 0:
            raise ValueError("cooldown must be >= 0")
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None
        self._trial_in_flight = False

    @property
    def enabled(self) -> bool:
        return self.failure_threshold > 0

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._opened_at is not None

    def check(self) -> bool:
        """Raise :class:`CircuitOpen` if a call must not be made right now.

        Returns ``True`` when this caller holds the half-open trial. A caller
        that does must end it with :meth:`record_success`,
        :meth:`record_failure` or, when the call ended without a verdict
        (cancelled, interrupted), :meth:`release_trial`.
        """
        if not self.enabled:
            return False
        with self._lock:
            if self._opened_at is None:
                return False
            elapsed = self._clock() - self._opened_at
            if elapsed < self.cooldown:
                raise CircuitOpen(self._failures, self.cooldown - elapsed)
            # Cooldown over: exactly one caller gets to try the backend.
            if self._trial_in_flight:
                raise CircuitOpen(self._failures, 0.0)
            self._trial_in_flight = True
            return True

    def release_trial(self) -> None:
        """Give back a trial whose call ended without a verdict.

        A cancelled or interrupted call says nothing about the backend, so it
        is not a failure; but the trial must be freed, or every later call is
        refused for as long as the process runs. The circuit stays open and
        the next caller gets the trial.
        """
        with self._lock:
            self._trial_in_flight = False

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._trial_in_flight = False

    def record_failure(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._failures += 1
            self._trial_in_flight = False
            if self._failures >= self.failure_threshold:
                self._opened_at = self._clock()

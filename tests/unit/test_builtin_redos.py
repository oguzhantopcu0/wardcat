"""Built-in regex patterns must not be ReDoS-prone.

The config loader already rejects catastrophic *custom* patterns; this does the
same for the library's own built-in patterns, which were previously trusted
without a check. Each compiled pattern is run against several adversarial inputs
under a timeout — catastrophic backtracking would exceed it.
"""

from __future__ import annotations

import concurrent.futures

from wardcat.detectors.regex_detector import _COMPILED

# Inputs designed to trigger backtracking in address/name/number-style patterns.
_ADVERSARIAL = [
    "a" * 400 + "!",
    "1" * 400 + "x",
    ("Ab " * 120) + "!",  # capitalized-word runs (Turkish ADDRESS style)
    ("Caddesi " * 80) + "x",
    (" " * 400),
    ("A" * 200) + ("0" * 200),
    ("4111 " * 120) + "z",  # card-like groups
]


def _finishes_fast(pattern, text, timeout: float = 1.5) -> bool:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(pattern.search, text)
        try:
            fut.result(timeout=timeout)
            return True
        except concurrent.futures.TimeoutError:
            return False


def test_builtin_patterns_are_redos_safe():
    for name, pattern in _COMPILED.items():
        for text in _ADVERSARIAL:
            assert _finishes_fast(pattern, text), (
                f"built-in pattern {name!r} did not finish in time on adversarial input "
                f"{text[:24]!r}… — possible ReDoS."
            )

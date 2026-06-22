"""Smoke tests that actually run the offline example scripts.

These guard against the examples silently rotting when the API changes (e.g.
the move to opt-in detection). Only the examples that need no external service
are run here; LLM/web examples are excluded.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

# (script, a substring that MUST appear in stdout to prove detection works)
OFFLINE_EXAMPLES = [
    ("demo.py", "CREDIT_CARD"),
    ("batch_and_async.py", "EMAIL"),
]


@pytest.mark.parametrize("script,expected", OFFLINE_EXAMPLES)
def test_example_runs_and_detects(script: str, expected: str):
    path = EXAMPLES_DIR / script
    assert path.exists(), f"example missing: {path}"
    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"{script} exited {result.returncode}\n{result.stderr}"
    # The whole point of the examples is detection — output must show a hit, not
    # an all-"clean" run (which is how opt-in silently broke batch_and_async.py).
    assert expected in result.stdout, (
        f"{script} produced no '{expected}' detection — did it stop detecting?\n{result.stdout}"
    )

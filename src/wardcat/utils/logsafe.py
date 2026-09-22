"""What a log line may say about a value: its length, never the value.

Every detector sees raw PII, and a debug line that quotes a rejected match or a
model's reply writes that PII to whatever the application logs to. The library
therefore logs :func:`describe` output in place of any value, at every level.
A salted digest was considered and rejected: low-entropy values (phone numbers,
national IDs) can be brute-forced from a digest by anyone holding the salt.
"""

from __future__ import annotations


def describe(value: str) -> str:
    """A PII-free stand-in for *value* in a log message: ``"len=12"``."""
    return f"len={len(value)}"

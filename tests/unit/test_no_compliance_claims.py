"""Nothing in the package or its docs claims compliance with a regime.

wardcat is a best-effort detector; it cannot make a system compliant with
KVKK, GDPR, HIPAA or PCI DSS, and saying so would mislead the people who rely
on it. This test fails on any positive claim in the README, the changelog, the
docs or the source, in English or Turkish. Negative statements ("does not by
itself make a system compliant", "KVKK uyumluluğu incelemesinin yerini
tutmaz") pass.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_CLAIMS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:KVKK|GDPR)(?:'?[a-zçğıöşü]*)?\s+(?:ile\s+)?(?:tam\s+)?(?:uyumlu|uygun)\b",
        r"\bcompliant\s+with\s+(?:the\s+)?(?:KVKK|GDPR|HIPAA|PCI)",
        r"\b(?:KVKK|GDPR|HIPAA|PCI[\s-]?DSS)[\s-]+compliant\b",
    )
]

SHOULD_MATCH = [
    "KVKK uyumlu",
    "KVKK'ya uyumlu",
    "KVKK'ya uygun",
    "KVKK ile uyumlu",
    "GDPR uyumlu",
    "compliant with GDPR",
    "GDPR-compliant",
    "HIPAA compliant",
    "PCI DSS compliant",
]
SHOULD_PASS = [
    "KVKK uyumluluğu incelemesinin yerini tutmaz",
    "not a substitute for compliance review (e.g. GDPR/KVKK)",
    "using it does not by itself make a system compliant",
    "no compliance claimed",
]


def _claims(text: str) -> list[str]:
    return [m.group(0) for rx in _CLAIMS for m in rx.finditer(text)]


@pytest.mark.parametrize("phrase", SHOULD_MATCH)
def test_the_patterns_catch_a_claim(phrase: str) -> None:
    assert _claims(phrase), phrase


@pytest.mark.parametrize("phrase", SHOULD_PASS)
def test_the_patterns_let_a_disclaimer_through(phrase: str) -> None:
    assert not _claims(phrase), phrase


def _files() -> list[Path]:
    return [
        ROOT / "README.md",
        ROOT / "CHANGELOG.md",
        *sorted((ROOT / "docs").rglob("*.md")),
        *sorted((ROOT / "src").rglob("*.py")),
    ]


def test_no_file_claims_compliance() -> None:
    found = [
        f"{path.relative_to(ROOT)}:{n}: {claim!r}"
        for path in _files()
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        for claim in _claims(line)
    ]
    assert found == [], "compliance claim(s) found:\n" + "\n".join(found)

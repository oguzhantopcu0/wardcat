"""Shared test fixtures.

As of 0.4.0 the guard starts **empty** (entities are opt-in). Many tests predate
that change and assume the old "everything on by default" policy. The
``legacy_guard`` fixture returns a factory that reproduces the pre-0.4.0 default
policy so those tests keep their original intent without re-listing entities.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from wardcat import Wardcat

# entity → action policy wardcat enabled by default before 0.4.0 made detection
# opt-in. (ORG, FINANCIAL_AMOUNT, SPECIAL_CATEGORY were off back then.)
LEGACY_POLICY: dict[str, str] = {
    "CREDIT_CARD": "hash",
    "EMAIL": "warn",
    "PHONE": "warn",
    "IBAN": "hash",
    "IP_ADDRESS": "warn",
    "IPv6": "warn",
    "TC_ID": "hash",
    "ADDRESS": "warn",
    "POSTAL_CODE": "warn",
    "UUID": "warn",
    "SSN": "hash",
    "MAC_ADDRESS": "warn",
    "JWT": "hash",
    "NIN": "hash",
    "DATE_OF_BIRTH": "hash",
    "VEHICLE_PLATE": "warn",
    "VAT_NUMBER": "warn",
    "PERSON": "hash",
    "UK_POSTAL_CODE": "warn",
    "US_ZIP_CODE": "warn",
    "EU_NATIONAL_ID": "warn",
    "PASSPORT": "warn",
    "CODICE_FISCALE": "warn",
    "CUSTOM_SECRET": "warn",
}


def make_legacy_guard(**kwargs) -> Wardcat:
    """Build an ``Wardcat`` pre-loaded with the pre-0.4.0 default entity policy."""
    guard = Wardcat(**kwargs)
    guard.add_entities(LEGACY_POLICY)
    return guard


@pytest.fixture
def legacy_guard() -> Callable[..., Wardcat]:
    """Factory fixture: ``legacy_guard(use_ner=False)`` → guard with the old policy."""
    return make_legacy_guard

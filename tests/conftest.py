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


# NER is configured via the with_ner() builder now, not constructor kwargs. The
# factory still accepts the old keyword style so the many legacy call sites keep
# working — it just routes them to with_ner().
_NER_KEYS = {"language", "spacy_model", "spacy_size", "spacy_auto_download"}


def make_legacy_guard(**kwargs) -> Wardcat:
    """Build an ``Wardcat`` pre-loaded with the pre-0.4.0 default entity policy.

    Accepts the historical NER keyword arguments (``use_ner``, ``spacy_model``,
    ``language``, …) and forwards them to :meth:`Wardcat.with_ner`.
    """
    ner_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _NER_KEYS}
    use_ner = kwargs.pop("use_ner", None)
    guard = Wardcat(**kwargs)
    if ner_kwargs or use_ner:
        if "spacy_auto_download" in ner_kwargs:
            ner_kwargs["auto_download"] = ner_kwargs.pop("spacy_auto_download")
        guard = guard.with_ner(**ner_kwargs)
    guard.add_entities(LEGACY_POLICY)
    return guard


@pytest.fixture
def legacy_guard() -> Callable[..., Wardcat]:
    """Factory fixture: ``legacy_guard(use_ner=False)`` → guard with the old policy."""
    return make_legacy_guard

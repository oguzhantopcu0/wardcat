"""Starting policies modelled on data-protection regimes.

A preset is an entity → action mapping, nothing more: it neither claims
compliance with the regulation it is named after nor switches any layer on.
Names and organisations need the NER or LLM layer, special-category data the
LLM layer; a preset that lists them says so in ``needs_layers``, and a guard
without that layer reports the entity as uncovered, as it does for any entity.

Use one as a base and adjust::

    guard = Wardcat(salt="s").with_preset("kvkk").with_ner(language="tr")
    guard.change_entity_action(Entity.PHONE, Action.REDACT)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from wardcat.core.registry import NER_ENTITIES, REGEX_ENTITIES
from wardcat.exceptions import ConfigError


@dataclass(frozen=True)
class Preset:
    """One starting policy."""

    name: str
    entities: Mapping[str, str]
    """Entity type → action."""
    needs_layers: frozenset[str]
    """Layers some of these entities can only be found by: ``"ner"``, ``"llm"``."""
    covers: str
    """One line on what the preset is for."""
    not_covered: str
    """What it does not do, so nobody reads a preset as a compliance guarantee."""


def _preset(name: str, entities: dict[str, str], covers: str, not_covered: str) -> Preset:
    needs: set[str] = set()
    for entity in entities:
        if entity in REGEX_ENTITIES:
            continue
        needs.add("ner" if entity in NER_ENTITIES else "llm")
    return Preset(
        name=name,
        entities=MappingProxyType(dict(entities)),
        needs_layers=frozenset(needs),
        covers=covers,
        not_covered=not_covered,
    )


PRESETS: dict[str, Preset] = {
    p.name: p
    for p in (
        _preset(
            "kvkk",
            {
                "TC_ID": "hash",
                "PERSON": "hash",
                "PHONE": "mask",
                "EMAIL": "mask",
                "ADDRESS": "redact",
                "IBAN": "hash",
                "CREDIT_CARD": "hash",
                "DATE_OF_BIRTH": "redact",
                "VEHICLE_PLATE": "hash",
                "SPECIAL_CATEGORY": "redact",
            },
            "Personal data as Turkey's KVKK defines it: identity, contact, financial "
            "and special-category data, with the identifiers Turkish text carries.",
            "Consent, retention, lawful basis and the rest of the law; organisation "
            "names; health data written without an identifiable person.",
        ),
        _preset(
            "gdpr",
            {
                "PERSON": "hash",
                "PHONE": "mask",
                "EMAIL": "mask",
                "ADDRESS": "redact",
                "IBAN": "hash",
                "CREDIT_CARD": "hash",
                "DATE_OF_BIRTH": "redact",
                "EU_NATIONAL_ID": "hash",
                "CODICE_FISCALE": "hash",
                "NIN": "hash",
                "PASSPORT": "hash",
                "NRP": "redact",
                "LOCATION": "warn",
                "SPECIAL_CATEGORY": "redact",
            },
            "Personal and Article 9 data across the EU and UK identifier schemes wardcat knows.",
            "Anything the regulation asks of the controller rather than of the text; "
            "national IDs outside the schemes listed; locations are reported, not "
            "replaced.",
        ),
        _preset(
            "pci_dss",
            {
                "CREDIT_CARD": "hash",
                "IBAN": "hash",
                "BANK_ROUTING": "hash",
                "CUSTOM_SECRET": "redact",
                "JWT": "redact",
            },
            "Cardholder and account data, and the credentials that guard it.",
            "Card-verification codes and expiry dates on their own; magnetic-stripe "
            "data; the network and process controls PCI DSS is mostly about.",
        ),
        _preset(
            "hipaa_lite",
            {
                "PERSON": "hash",
                "DATE_OF_BIRTH": "redact",
                "SSN": "hash",
                "PHONE": "mask",
                "EMAIL": "mask",
                "ADDRESS": "redact",
                "US_ZIP_CODE": "warn",
                "IP_ADDRESS": "hash",
                "NHS_NUMBER": "hash",
                "SPECIAL_CATEGORY": "redact",
            },
            "The direct identifiers of the Safe Harbor list that appear in free text.",
            "Medical record and account numbers, device serials, biometric and photo "
            "data, dates other than birth; nothing here makes text de-identified in "
            "the legal sense.",
        ),
        _preset(
            "secrets_only",
            {
                "CUSTOM_SECRET": "redact",
                "JWT": "redact",
                "CRYPTO_WALLET": "hash",
                "USERNAME": "hash",
            },
            "Credentials and tokens in logs, tickets and prompts; no personal data.",
            "Every kind of personal data; secrets with no known prefix and no keyword beside them.",
        ),
    )
}


def supported_presets() -> tuple[str, ...]:
    """The preset names, in the order they are documented."""
    return tuple(PRESETS)


def get_preset(name: str) -> Preset:
    """Look a preset up by name.

    :raises ConfigError: for a name that is not one of :func:`supported_presets`.
    """
    try:
        return PRESETS[name]
    except KeyError:
        raise ConfigError(
            f"Unknown preset {name!r}. Available presets: {', '.join(PRESETS)}."
        ) from None

"""
Predefined entity group sets for common use cases.

Groups entities by language/region or purpose so users don't have to
enumerate individual entity types:

    guard = AIGuard(use_ner=False)
    for entity in european_entities():
        guard.add_entity(entity, action="hash")

Available groups:
    core_entities()         — universal PII (email, phone, credit card, IBAN)
    financial_entities()    — credit card, IBAN, SSN, bank/financial IDs
    turkish_entities()      — TR-specific: TC_ID, POSTAL_CODE
    european_entities()     — EU: EU_NATIONAL_ID, CODICE_FISCALE, NIN, IBAN, PASSPORT
    uk_entities()           — UK: NIN, UK_POSTAL_CODE, PASSPORT
    us_entities()           — US: SSN, US_ZIP_CODE, PASSPORT
    network_entities()      — IP_ADDRESS, IPv6, MAC_ADDRESS, UUID, JWT
    identity_entities()     — PASSPORT, NIN, SSN, TC_ID, EU_NATIONAL_ID, CODICE_FISCALE
    all_entities()          — all known entity types
"""

from __future__ import annotations

from ai_guard.core.models import KNOWN_ENTITY_TYPES


def core_entities() -> frozenset[str]:
    """Universal PII: email, phone, credit card, IBAN."""
    return frozenset({"EMAIL", "PHONE", "CREDIT_CARD", "IBAN"})


def financial_entities() -> frozenset[str]:
    """Financial identifiers: credit card, IBAN, SSN, JWT."""
    return frozenset({"CREDIT_CARD", "IBAN", "SSN", "JWT"})


def turkish_entities() -> frozenset[str]:
    """Turkey-specific entities: TC_ID, POSTAL_CODE, VEHICLE_PLATE, plus core."""
    return core_entities() | frozenset({"TC_ID", "POSTAL_CODE", "VEHICLE_PLATE"})


def european_entities() -> frozenset[str]:
    """EU/European entities: EU_NATIONAL_ID, CODICE_FISCALE, NIN, IBAN, PASSPORT."""
    return core_entities() | frozenset(
        {
            "EU_NATIONAL_ID",
            "CODICE_FISCALE",
            "NIN",
            "PASSPORT",
            "UK_POSTAL_CODE",
        }
    )


def uk_entities() -> frozenset[str]:
    """UK-specific entities: NIN, UK_POSTAL_CODE, PASSPORT."""
    return core_entities() | frozenset({"NIN", "UK_POSTAL_CODE", "PASSPORT"})


def us_entities() -> frozenset[str]:
    """US-specific entities: SSN, US_ZIP_CODE, PASSPORT."""
    return core_entities() | frozenset({"SSN", "US_ZIP_CODE", "PASSPORT"})


def network_entities() -> frozenset[str]:
    """Network / technical identifiers: IP, IPv6, MAC, UUID, JWT."""
    return frozenset({"IP_ADDRESS", "IPv6", "MAC_ADDRESS", "UUID", "JWT"})


def identity_entities() -> frozenset[str]:
    """Government-issued identity documents and numbers."""
    return frozenset(
        {
            "PASSPORT",
            "NIN",
            "SSN",
            "TC_ID",
            "EU_NATIONAL_ID",
            "CODICE_FISCALE",
        }
    )


def all_entities() -> frozenset[str]:
    """All known entity types (regex + NER)."""
    return frozenset(KNOWN_ENTITY_TYPES)

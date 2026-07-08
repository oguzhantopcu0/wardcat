"""Static detector/entity registry — which entity type each layer can detect.

Shared by the entity-policy layer (deciding where an entity is enabled) and the
detector-build path (deciding what each detector scans for). Kept separate so
both can import it without depending on ``Wardcat``.
"""

from __future__ import annotations

from wardcat.llm.prompt import SUPPORTED_ENTITIES as _LLM_ENTITIES

# Entity types each detector layer can produce.
REGEX_ENTITIES: frozenset[str] = frozenset(
    {
        "CREDIT_CARD",
        "EMAIL",
        "PHONE",
        "IBAN",
        "IP_ADDRESS",
        "IPv6",
        "TC_ID",
        "ADDRESS",
        "POSTAL_CODE",
        "UUID",
        "SSN",
        "MAC_ADDRESS",
        "JWT",
        "NIN",
        "CUSTOM_SECRET",
        "UK_POSTAL_CODE",
        "US_ZIP_CODE",
        "EU_NATIONAL_ID",
        "PASSPORT",
        "CODICE_FISCALE",
        "DATE_OF_BIRTH",
        "VEHICLE_PLATE",
        "FINANCIAL_AMOUNT",
        "VAT_NUMBER",
    }
)
NER_ENTITIES: frozenset[str] = frozenset({"PERSON", "ORG", "ADDRESS"})

# Which detector layers a filter can be applied to, and which entities each supports.
LAYER_ENTITIES: dict[str, frozenset[str]] = {
    "regex": REGEX_ENTITIES,
    "ner": NER_ENTITIES,
    "llm": _LLM_ENTITIES,
}
VALID_LAYERS: frozenset[str] = frozenset(LAYER_ENTITIES)

"""Static detector/entity registry — which entity type each layer can detect.

Shared by the entity-policy layer (deciding where an entity is enabled) and the
detector-build path (deciding what each detector scans for). Kept separate so
both can import it without depending on ``AIGuard``.
"""

from __future__ import annotations

from ai_guard.llm.prompt import SUPPORTED_ENTITIES as _LLM_ENTITIES

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

# GLiNER (zero-shot transformer NER) label → ai-guard entity type.
# The keys are the labels the GLiNER2 model is asked to extract; the values are
# the ai-guard entity types they map onto. GLiNER is a probabilistic layer, so
# these spans get sub-1.0 confidence — a checksum-validated regex span always
# wins an overlap (see DetectionEngine._resolve_overlaps). Labels with no clean
# ai-guard counterpart (drivers_license, card_cvv, username, …) are left out on
# purpose rather than force-mapped.
GLINER_LABEL_MAP: dict[str, str] = {
    "person": "PERSON",
    "full_name": "PERSON",
    "first_name": "PERSON",
    "middle_name": "PERSON",
    "last_name": "PERSON",
    "date_of_birth": "DATE_OF_BIRTH",
    "email": "EMAIL",
    "phone_number": "PHONE",
    "address": "ADDRESS",
    "street_address": "ADDRESS",
    "city": "ADDRESS",
    "state_or_region": "ADDRESS",
    "country": "ADDRESS",
    "postal_code": "POSTAL_CODE",
    "national_id_number": "EU_NATIONAL_ID",
    "government_id": "EU_NATIONAL_ID",
    "passport_number": "PASSPORT",
    "tax_id": "VAT_NUMBER",
    "tax_number": "VAT_NUMBER",
    "iban": "IBAN",
    "payment_card": "CREDIT_CARD",
    "card_number": "CREDIT_CARD",
    "ip_address": "IP_ADDRESS",
    "password": "CUSTOM_SECRET",
    "secret": "CUSTOM_SECRET",
    "api_key": "CUSTOM_SECRET",
    "access_token": "CUSTOM_SECRET",
    "recovery_code": "CUSTOM_SECRET",
}
GLINER_ENTITIES: frozenset[str] = frozenset(GLINER_LABEL_MAP.values())

# Which detector layers a filter can be applied to, and which entities each supports.
LAYER_ENTITIES: dict[str, frozenset[str]] = {
    "regex": REGEX_ENTITIES,
    "ner": NER_ENTITIES,
    "gliner": GLINER_ENTITIES,
    "llm": _LLM_ENTITIES,
}
VALID_LAYERS: frozenset[str] = frozenset(LAYER_ENTITIES)

"""
LLM PII tespit prompt'u.

Küçük modellerde (3B–8B) iyi sonuç için:
- Her entity tipi açıkça tanımlanmış
- Neyin PII OLMADIĞI örneklenmiş (false positive baskısı)
- Türkçe + İngilizce karma örnekler
- Yapısal kısıtlar ve bağlamsal ipuçları verilmiş
- temperature=0 ile deterministik
"""
from __future__ import annotations

# Entity açıklamaları: modele ne araması gerektiğini öğretir.
_ENTITY_DESCRIPTIONS: dict[str, str] = {
    "ORG":           "name of a specific company, institution, or organization that could identify a person in context (e.g. 'Acme Corp', 'İş Bankası', 'Google LLC')",
    "PERSON":        "full name of a real person (e.g. 'Ali Veli', 'John Smith', 'Mehmet Demir')",
    "EMAIL":         "email address (e.g. 'user@example.com')",
    "PHONE":         "phone number including country/area code (e.g. '0532 123 4567', '+90 533 987 6543')",
    "TC_ID":         "Turkish national ID — exactly 11 digits starting with non-zero (e.g. '12345678901')",
    "IBAN":          "IBAN bank account number starting with 2-letter country code (e.g. 'TR33 0006 1005 1978 6457 8413 26')",
    "CREDIT_CARD":   "credit/debit card number, 13–19 digits, may have spaces or dashes (e.g. '4111 1111 1111 1111')",
    "IP_ADDRESS":    "IPv4 address (e.g. '192.168.1.1')",
    "POSTAL_CODE":   "postal/ZIP code (e.g. '34100')",
    "ADDRESS":       "physical street address including street name and number (e.g. 'Atatürk Cad. No:5')",
    "UUID":          "UUID / GUID identifier (e.g. '550e8400-e29b-41d4-a716-446655440000')",
    "SSN":           "US Social Security Number — format: 3-2-4 digits with dashes (e.g. '123-45-6789')",
    "MAC_ADDRESS":   "network hardware (MAC) address — colon or dash separated hex pairs (e.g. '00:1A:2B:3C:4D:5E')",
    "JWT":           "JSON Web Token — three base64url segments separated by dots, starts with 'eyJ'",
    "IPv6":          "IPv6 network address (e.g. '2001:db8::8a2e:0370:7334')",
    "NIN":           "UK National Insurance Number — two letters, 6 digits, one letter A-D (e.g. 'AB123456C')",
    "UK_POSTAL_CODE": "British postcode — area code + space + sector + unit letters (e.g. 'SW1A 1AA', 'EC1A 1BB', 'GU21 6TH')",
    "US_ZIP_CODE":   "US ZIP+4 postal code — 5 digits, dash, 4 digits (e.g. '10001-1234', '90210-3456')",
    "EU_NATIONAL_ID": (
        "European national identity number — "
        "Spanish DNI: 8 digits + check letter (e.g. '12345678Z'), "
        "Spanish NIE: X/Y/Z + 7 digits + check letter (e.g. 'X1234567L'), "
        "French INSEE: 15-digit social security number, "
        "German Personalausweis or Steuer-ID"
    ),
    "PASSPORT":      (
        "passport number of any country — typically 1-2 capital letters followed by 6-9 digits "
        "(e.g. US: 'A12345678', UK: '123456789', German: 'C01X00T47', French: '06AB12345'). "
        "Only extract when clearly labeled as a passport number."
    ),
    "CUSTOM_SECRET": (
        "contextual secret/credential signaled by a keyword prefix such as: "
        "şifre=, password=, pass=, pwd=, api_key=, api-key=, apikey=, token=, "
        "secret=, sk-, key=, erişim kodu, access code, auth (e.g. 'db_pass=S3cr3t!42', "
        "'sk-prod-xK92mNzL8qW3', 'ALPHA-BRAVO-42' after 'erişim kodu'). "
        "Extract ONLY the secret value, not the keyword."
    ),
}

_SYSTEM_TEMPLATE = """\
You are a PII (Personally Identifiable Information) detection engine.

TASK
====
Find every piece of sensitive information in the text below.
Requested entity types and their definitions:

{entity_definitions}

OUTPUT RULES
============
- Return ONLY a valid JSON array — no markdown, no explanation, nothing else.
- Copy the EXACT text as it appears in the input — do not paraphrase or modify.
- Each item: {{"type": "ENTITY_TYPE", "text": "exact matched text"}}
- If nothing is found → return: []

DO NOT extract:
- Numbers that are NOT identifiers (prices, percentages, quantities, dates, order IDs)
- Generic words, city names, or company names unless they are part of a personal name
- Common Turkish or English nouns/adjectives — these are NOT person names:
  "hedef", "müşteri", "destek", "proje", "sistem", "ekip", "kullanıcı",
  "yönetici", "çeyrek", "dönem", "rapor", "görev", "kayıt", "konu", "durum"
- Timestamps, durations, or meeting times
- Role/occupation titles alone without an accompanying proper name
  (e.g. "Müdür", "Temsilci", "Kullanıcı" without a first+last name next to them)
- PERSON must be a proper human name (first name + last name or clearly a personal name).
  Single common words are NEVER a person name.

EXAMPLES
========
Input: "Müşterimiz Ali Veli (ali.veli@sirket.com, TC: 12345678901) 4111111111111111 kartıyla ödedi. Tel: 0532 123 4567"
Output: [{{"type":"PERSON","text":"Ali Veli"}},{{"type":"EMAIL","text":"ali.veli@sirket.com"}},{{"type":"TC_ID","text":"12345678901"}},{{"type":"CREDIT_CARD","text":"4111111111111111"}},{{"type":"PHONE","text":"0532 123 4567"}}]

Input: "Dear John Smith, your IBAN TR33 0006 1005 1978 6457 8413 26 is confirmed. Contact: john@acme.com"
Output: [{{"type":"PERSON","text":"John Smith"}},{{"type":"IBAN","text":"TR33 0006 1005 1978 6457 8413 26"}},{{"type":"EMAIL","text":"john@acme.com"}}]

Input: "Veritabanı şifresi db_pass=S3cr3t!42 — kimseyle paylaşmayın."
Output: [{{"type":"CUSTOM_SECRET","text":"S3cr3t!42"}}]

Input: "API anahtarı: sk-prod-xK92mNzL8qW3 — sadece prod ortamında kullanın."
Output: [{{"type":"CUSTOM_SECRET","text":"sk-prod-xK92mNzL8qW3"}}]

Input: "Toplantı erişim kodu ALPHA-BRAVO-42 onaylandı."
Output: [{{"type":"CUSTOM_SECRET","text":"ALPHA-BRAVO-42"}}]

Input: "Sipariş ID: 98765, adet: 3, toplam: 450 TL."
Output: []

Input: "Hava bugün çok güzel, piknik yapalım."
Output: []

Input: "Device MAC: 00:1A:2B:3C:4D:5E, IPv6: 2001:db8::1, session: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.abc123"
Output: [{{"type":"MAC_ADDRESS","text":"00:1A:2B:3C:4D:5E"}},{{"type":"IPv6","text":"2001:db8::1"}},{{"type":"JWT","text":"eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.abc123"}}]

Input: "User UUID: 550e8400-e29b-41d4-a716-446655440000, SSN: 123-45-6789, NIN: AB123456C"
Output: [{{"type":"UUID","text":"550e8400-e29b-41d4-a716-446655440000"}},{{"type":"SSN","text":"123-45-6789"}},{{"type":"NIN","text":"AB123456C"}}]

Input: "Toplantı saat 14:00'te 3. katta."
Output: []
"""

_USER_TEMPLATE = """\
Text:
\"\"\"{text}\"\"\"

JSON:"""


def build_messages(text: str, entity_types: set[str]) -> list[dict]:
    """Build system + user messages for chat-capable backends (Transformers, OpenAI chat).

    Args:
        text: The input text to scan for PII.
        entity_types: Set of entity type names to detect.

    Returns:
        A list of ``{"role": ..., "content": ...}`` dicts suitable for
        ``tokenizer.apply_chat_template()`` or the OpenAI messages API.
    """
    lines = []
    for etype in sorted(entity_types):
        desc = _ENTITY_DESCRIPTIONS.get(etype, f"sensitive data of type {etype}")
        lines.append(f"  {etype}: {desc}")
    entity_definitions = "\n".join(lines)

    system = _SYSTEM_TEMPLATE.format(entity_definitions=entity_definitions)
    user   = _USER_TEMPLATE.format(text=text)
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


def build_prompt(text: str, entity_types: set[str]) -> str:
    """Build the full system+user prompt for LLM PII detection.

    Args:
        text: The input text to scan for PII.
        entity_types: Set of entity type names to look for (e.g. ``{"EMAIL", "PHONE"}``).

    Returns:
        A single string combining the system instructions and user turn,
        ready to be sent to the LLM as a plain-text prompt.
    """
    lines = []
    for etype in sorted(entity_types):
        desc = _ENTITY_DESCRIPTIONS.get(etype, f"sensitive data of type {etype}")
        lines.append(f"  {etype}: {desc}")
    entity_definitions = "\n".join(lines)

    system = _SYSTEM_TEMPLATE.format(entity_definitions=entity_definitions)
    user   = _USER_TEMPLATE.format(text=text)
    return f"{system}\n\n{user}"

# Presets

A preset is a starting policy: an entity → action mapping modelled on a
data-protection regime. It is a shortcut for a long `add_entities()` call and
nothing more — it switches no detection layer on, and it claims no compliance
with the regulation it is named after. Read each preset's *not covered* line
before relying on it.

```python
from wardcat import Wardcat, Entity, Action

guard = Wardcat(salt="s").with_preset("kvkk").with_ner(language="tr")
guard.change_entity_action(Entity.PHONE, Action.REDACT)   # adjust afterwards
guard.remove_entity(Entity.VEHICLE_PLATE)

Wardcat.supported_presets()
```

```yaml
preset: kvkk            # the base
entities:               # the file's own entries win over the preset
  PHONE: { enabled: true, action: redact }
```

Names and organisations are found by the NER or LLM layer, special-category
data by the LLM layer. A preset that lists them needs that layer; without it
the entity is reported as uncovered at the first scan, as any entity is.

The actions a preset picks follow one rule: an identifier that a record may
need to be linked on again is `hash`ed (stable, salted), contact details that
still need to be readable are `mask`ed, and anything with no downstream use is
`redact`ed.

## `kvkk`

Personal data as Turkey's KVKK defines it: identity, contact, financial and
special-category data, with the identifiers Turkish text carries.

`TC_ID` → `hash` · `PERSON` → `hash` · `PHONE` → `mask` · `EMAIL` → `mask` ·
`ADDRESS` → `redact` · `IBAN` → `hash` · `CREDIT_CARD` → `hash` ·
`DATE_OF_BIRTH` → `redact` · `VEHICLE_PLATE` → `hash` · `SPECIAL_CATEGORY` → `redact`

Needs `ner` (or `llm`) for names, `llm` for special-category data.

**Not covered:** consent, retention, lawful basis and the rest of the law;
organisation names; health data written without an identifiable person.

## `gdpr`

Personal and Article 9 data across the EU and UK identifier schemes wardcat
knows.

`PERSON` → `hash` · `PHONE` → `mask` · `EMAIL` → `mask` · `ADDRESS` → `redact` ·
`IBAN` → `hash` · `CREDIT_CARD` → `hash` · `DATE_OF_BIRTH` → `redact` ·
`EU_NATIONAL_ID` → `hash` · `CODICE_FISCALE` → `hash` · `NIN` → `hash` ·
`PASSPORT` → `hash` · `NRP` → `redact` · `LOCATION` → `warn` ·
`SPECIAL_CATEGORY` → `redact`

Needs `ner` (or `llm`) for names, nationalities and places, `llm` for
special-category data.

**Not covered:** anything the regulation asks of the controller rather than of
the text; national IDs outside the schemes listed; locations are reported, not
replaced.

## `pci_dss`

Cardholder and account data, and the credentials that guard it.

`CREDIT_CARD` → `hash` · `IBAN` → `hash` · `BANK_ROUTING` → `hash` ·
`CUSTOM_SECRET` → `redact` · `JWT` → `redact`

Regex only; no model needed.

**Not covered:** card-verification codes and expiry dates on their own;
magnetic-stripe data; the network and process controls PCI DSS is mostly about.

## `hipaa_lite`

The direct identifiers of the Safe Harbor list that appear in free text.

`PERSON` → `hash` · `DATE_OF_BIRTH` → `redact` · `SSN` → `hash` · `PHONE` → `mask` ·
`EMAIL` → `mask` · `ADDRESS` → `redact` · `US_ZIP_CODE` → `warn` ·
`IP_ADDRESS` → `hash` · `NHS_NUMBER` → `hash` · `SPECIAL_CATEGORY` → `redact`

Needs `ner` (or `llm`) for names, `llm` for special-category data.

**Not covered:** medical record and account numbers, device serials, biometric
and photo data, dates other than birth; nothing here makes text de-identified
in the legal sense.

## `secrets_only`

Credentials and tokens in logs, tickets and prompts; no personal data.

`CUSTOM_SECRET` → `redact` · `JWT` → `redact` · `CRYPTO_WALLET` → `hash` ·
`USERNAME` → `hash`

Regex only; no model needed.

**Not covered:** every kind of personal data; secrets with no known prefix and
no keyword beside them.

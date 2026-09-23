# Entity types

Every entity wardcat can detect, with the action it takes when you enable it
without naming one, and the layer that finds it. Detection is opt-in — a bare
`Wardcat()` detects nothing until you `add_entity(...)`. The same list at the
prompt: `wardcat entities [--layer regex|ner|llm]`; in code:
`Wardcat.supported_entities(layer)`.

## Regex (built-in, no dependencies)

### Financial & identity

| Entity | Default action | Description |
|---|---|---|
| `CREDIT_CARD` | `hash` | Visa (13/16/19-digit), MasterCard (`51`–`55` and the 2-series `2221`–`2720`), Amex, Discover, Diners, JCB (`3528`–`3589` and the legacy 15-digit `1800`/`2131`), Maestro, Troy — with or without separators; Luhn (mod-10) validated |
| `IBAN` | `hash` | International IBAN — mod-97 checksum validated |
| `SSN` | `hash` | US Social Security Number (123-45-6789), or written without dashes after its label |
| `BANK_ROUTING` | `hash` | US bank routing number (ABA / RTN) — mod-10 checksum over a Federal Reserve prefix |
| `CRYPTO_WALLET` | `hash` | Bitcoin (Base58Check and bech32/bech32m segwit, both checksum-verified) and Ethereum-style `0x` addresses |
| `NIN` | `hash` | UK National Insurance Number (AB123456C) |
| `NHS_NUMBER` | `hash` | UK NHS number — 10 digits, mod-11 checksum |
| `TC_ID` | `hash` | Turkish national ID — 11 digits, whole or grouped, Nüfus İdaresi checksum validated |
| `EU_NATIONAL_ID` | `hash` | Spanish DNI / NIE, French INSEE, Dutch BSN, Polish PESEL — each verified against its own check rule |
| `CODICE_FISCALE` | `hash` | Italian tax code (RSSMRA85T10A562S) |
| `VAT_NUMBER` | `warn` | EU VAT (DE/FR/GB/IT/ES/AT/NL prefixed) + Turkish Vergi No (keyword-based) |
| `PASSPORT` | `hash` | Passport numbers — keyword-based (`passport no:`, `pasaport`, `Reisepass`, `passeport`) |
| `DATE_OF_BIRTH` | `hash` | Birth dates — month names (TR/EN/DE/FR) or keyword + numeric/ISO date |
| `VEHICLE_PLATE` | `warn` | Turkish vehicle plates (34 ABC 123) |
| `FINANCIAL_AMOUNT` | `redact` | Monetary amounts (₺/$/€/£, `45.000 TL`, `2.1 milyon TL`) — **off by default** |

### Contact & location

| Entity | Default action | Description |
|---|---|---|
| `EMAIL` | `warn` | RFC-compliant email addresses |
| `PHONE` | `warn` | Turkish (`0`, `+90`), French national (`01 23 45 67 89`), German mobile (`0151 …`), international E.164 (`+1`, `+44`, …), and any national format once it is labelled (`Phone:`, `call me at …`); other regions via [`with_phone_regions()`](../guide/configuration.md#phone-regions) |
| `ADDRESS` | `warn` | Turkish (Cad., Sok., Mah.), English (Street, Avenue, Road…), French (Rue, Allée…), Spanish (Calle, Avenida…), Italian (Piazza, Corso…), Dutch (straat, gracht…), German (Straße, Weg, Platz…) |
| `POSTAL_CODE` | `warn` | Turkish postal codes (01000–81999) |
| `UK_POSTAL_CODE` | `warn` | British postcodes (SW1A 1AA, GU21 6TH, M1 1AE) |
| `US_ZIP_CODE` | `warn` | US ZIP+4 codes (12345-6789) |

### Network & technical

| Entity | Default action | Description |
|---|---|---|
| `IP_ADDRESS` | `warn` | IPv4 addresses — the quad must stand alone, so a longer dotted run (`03.93.92.16.85`, a French phone number) is not read as an address |
| `IPv6` | `warn` | IPv6 addresses (full and compressed forms) |
| `MAC_ADDRESS` | `warn` | Network hardware address (00:1A:2B:3C:4D:5E) |
| `IMEI` | `hash` | Mobile device IMEI — 15 digits, Luhn-checked |
| `UUID` | `warn` | RFC 4122 UUID / GUID |
| `USERNAME` | `hash` | Account name introduced by its keyword — `kullanıcı adı ahmet.yilmaz`, `username: jsmith42`, `login jdoe`. Only the handle is taken |
| `JWT` | `hash` | JSON Web Token (starts with `eyJ`) |
| `CUSTOM_SECRET` | `hash` | API keys & tokens: OpenAI/Anthropic (`sk-`, `sk-ant-`), Stripe (`sk_live_`), AWS (`AKIA`), Google (`AIza`, `ya29.`), GitHub (`ghp_`, fine-grained `github_pat_`), GitLab (`glpat-`), Slack (`xoxb-`, webhook URLs), Twilio (`SK`/`AC`), SendGrid (`SG.`), npm (`npm_`), Hugging Face (`hf_`), Shopify, DigitalOcean, Azure storage keys, Sentry DSNs, connection-string passwords and PEM private-key blocks. Also a credential written into a sentence — `parolası ise …`, `password is …`, `erişim kodu …` — where the word beside it is the only evidence; only the value is taken, not the keyword |
| `HIGH_ENTROPY_STRING` | `redact` | A run of 32+ base64-shaped characters with the Shannon entropy of a generated key, or a long hex digest — a secret with no known prefix and no keyword. **Off by default**, scored `0.70` under the default floor; see [the regex layer](../guide/layers.md#regex) |

## SpaCy NER (requires `wardcat[ner]` + a language model)

| Entity | Default action | Description |
|---|---|---|
| `PERSON` | `hash` | Person names (first + last) — cross-language |
| `ORG` | `warn` | Organization / company names |
| `ADDRESS` | `warn` | Street addresses and facilities (complements regex) |
| `LOCATION` | `warn` | Countries, cities, regions and geographic features (spaCy `GPE` / `LOC`). **These used to arrive as `ADDRESS`** — enable this to keep covering them |
| `NRP` | `redact` | Nationality, religious or political group — GDPR Art. 9 data. **These used to arrive as `ORG`.** Off by default: these are ordinary words |

!!! note "NER requires an explicit model — there is no default"
    NER is **off by default**; calling `with_ner()` without a model (or language)
    raises `ConfigError`. Choose a model in a documented way via `language=`
    (recommended) or `spacy_model=`. Running, say, the Turkish model on German
    text produces noisy results, so pick the model per language (or rely on the
    LLM layer for cross-language names). A multilingual gazetteer filters out job
    titles, HR terms, and abbreviations (EN/DE/FR/TR) that NER models commonly
    mislabel. How to pick a language, a size tier and several models at once is
    on the [detection layers](../guide/layers.md#spacy-ner-ner) page.

## LLM (requires an on-prem LLM backend)

| Entity | Default action | Description |
|---|---|---|
| `PASSPORT` | `hash` | Passport numbers of any country — contextual detection (e.g. `Passport: A12345678`) |
| `EU_NATIONAL_ID` | `hash` | French INSEE, German Personalausweis — contextual variants not caught by regex |
| `UK_POSTAL_CODE` | `warn` | Postcodes in ambiguous contexts |
| `US_ZIP_CODE` | `warn` | ZIP codes in ambiguous contexts |
| `CUSTOM_SECRET` | `hash` | Contextual secrets — `password=VALUE`, `api_key=VALUE`, access codes |
| `SPECIAL_CATEGORY` | `redact` | GDPR Art.9 special-category data — health, religion, ethnicity, political opinion, sexual orientation, trade-union, genetic/biometric. **Off by default** (semantic, LLM-only) |
| *(any above)* | — | LLM supplements and verifies all regex/NER entity types |

!!! note "GDPR special categories"
    `SPECIAL_CATEGORY` flags sensitive statements that have no pattern — a stated
    health condition, religious or political affiliation, or trade-union
    membership — which only the LLM can detect. It is off by default; enable it
    under `llm_detector.entities`. Because it is semantic and subjective, expect
    lower precision than structural entities.

`with_llm()` brings its own default policy — most of these types switched on,
each with the action above — so a guard with the LLM layer detects more than
the entities you named. The [configuration guide](../guide/configuration.md#the-llm-layers-own-entity-policy)
shows how to take control of it.

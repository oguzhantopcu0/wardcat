```
    _    ___        ____  _   _    _    ____  ____
   / \  |_ _|      / ___|| | | |  / \  |  _ \|  _ \
  / _ \  | |  ___ | |  _ | | | | / _ \ | |_) | | | |
 / ___ \ | | |___|| |_| || |_| |/ ___ \|  _ <| |_| |
/_/   \_|___|      \____| \___//_/   \_|_| \_|____/
```

**PII detection and anonymization for LLM inputs** — hybrid regex + NER + on-prem LLM engine.

`ai-guard` scans text for personally identifiable information (PII) before it reaches an LLM, and either warns about or replaces the sensitive data with salted SHA-256 hashes. It supports Turkish, English, German, and French out of the box.

```python
from ai_guard import LLMGuard

guard = (
    LLMGuard(salt="my-secret-salt")
    .configure_entity("CREDIT_CARD", enabled=True, action="hash")
    .configure_entity("EMAIL",       enabled=True, action="warn")
    .configure_entity("TC_ID",       enabled=True, action="hash")
)

result = guard.scan("Name: Ali Veli, card: 4532 0151 1283 0366, email: ali@example.com")
print(result.sanitized_text)
# Name: Ali Veli, card: [CREDIT_CARD:ea782818c5a992a8], email: ali@example.com
```

---

## Features

- **Hybrid detection** — Regex + SpaCy NER + on-prem LLM (Ollama, OpenAI-compatible, HuggingFace Transformers)
- **Ensemble adjudication** (optional) — the LLM verifies/relabels/drops regex & NER candidates and adds what they missed, in one call; deterministic regex results are always protected
- **Four actions** — `warn` (keep text, report only), `hash` (`[TYPE:16hex]` via SHA-256 + salt), `redact` (`[TYPE]` label, no hash), `mask` (entity-aware partial masking)
- **Checksum validation** — TC_ID (Nüfus İdaresi algorithm), IBAN (mod-97), and CREDIT_CARD (Luhn mod-10) validated before flagging — eliminates false positives
- **Rainbow table protection** — user-defined salt for all hashes
- **Two APIs** — method chaining (programmatic) and YAML (declarative)
- **CLI** — `ai-guard scan`, `ai-guard batch`, `ai-guard models`
- **Multilingual support** — Turkish, English, German, and French for names, addresses, birth dates, and phone numbers; plus Spanish, Italian, Dutch address patterns; TC_ID, IBAN, SSN, NIN, DNI/NIE, UK postcodes, US ZIP+4, EU VAT numbers and more
- **Secret detection** — API keys and tokens (OpenAI, Anthropic, Stripe, AWS, Google, GitHub, GitLab, Slack, Twilio, SendGrid, npm) and PEM private keys
- **Passport detection** — contextual passport number detection (regex keyword-based + LLM) for any country
- **DoS protection** — inputs exceeding 500 KB are rejected
- **Safe logging API** — `result.redacted()` returns a PII-free dict for logs and APIs

---

## Installation

```bash
# Base install — regex detection + Ollama/OpenAI-compatible LLM backend
pip install ai-guard

# + SpaCy NER (PERSON, ORG, ADDRESS detection)
pip install "ai-guard[ner]"

# + HuggingFace Transformers backend (local GPU/CPU inference)
pip install "ai-guard[transformers]"

# Everything at once
pip install "ai-guard[all]"
```

Or clone and install with [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/oguzhantopcu0/ai-guard.git
cd ai-guard
uv sync
```

To use SpaCy NER (PERSON, ORG, ADDRESS detection):

```bash
# List available models
uv run python -m ai_guard spacy list

# List Turkish models
uv run python -m ai_guard spacy list --lang tr

# Download English model (recommended)
uv run python -m ai_guard spacy download en_core_web_sm

# Download Turkish model (recommended)
uv run python -m ai_guard spacy download tr_core_news_md

# Check installed models
uv run python -m ai_guard spacy installed
```

> If the requested SpaCy model is not installed, ai-guard automatically falls back to any installed model of the same language and logs a warning. SpaCy is not required if you only need regex-based detection.

---

## Quick Start

### Programmatic API

```python
from ai_guard import LLMGuard

guard = (
    LLMGuard(salt="my-secret-salt")
    .configure_entity("CREDIT_CARD", enabled=True, action="hash")
    .configure_entity("EMAIL",       enabled=True, action="warn")
    .configure_entity("TC_ID",       enabled=True, action="hash")
)

result = guard.scan("""
  Customer: Ali Veli, TC: 12345678950
  Card: 4532 0151 1283 0366
  Email: ali.veli@example.com
""")

print(result.sanitized_text)
# Customer: Ali Veli, TC: [TC_ID:86349f34a1bc2d5e]
# Card: [CREDIT_CARD:ea782818c5a992a8]
# Email: ali.veli@example.com   ← warn: text is kept

for v in result.violations:
    print(f"[{v.action.value}] {v.entity_type}: {v.original!r}")
# [hash] TC_ID: '12345678950'
# [hash] CREDIT_CARD: '4532 0151 1283 0366'
# [warn] EMAIL: 'ali.veli@example.com'
```

### Declarative API (YAML)

```python
from ai_guard import LLMGuard

guard = LLMGuard(config_path="config/my_policy.yaml")
result = guard.scan(text)
```

```yaml
# config/my_policy.yaml
salt: ""          # read from env in production
spacy_model: "en_core_web_sm"
use_ner: true

entities:
  CREDIT_CARD: { enabled: true,  action: hash }
  EMAIL:       { enabled: true,  action: warn }
  TC_ID:       { enabled: true,  action: hash }
  IBAN:        { enabled: true,  action: hash }
  PERSON:      { enabled: true,  action: hash }
  ORG:         { enabled: false, action: warn }
```

### Batch Scanning

```python
results = guard.scan_batch([
    "ali@example.com",
    "Card: 4111 1111 1111 1111",
    "Clean text.",
])

for r in results:
    print(r.is_clean, len(r.violations))
# False 1
# False 1
# True  0
```

### On-prem LLM

Two options for LLM-based detection:

**Option 1 — Run locally with Ollama:**

```bash
# Install Ollama: https://ollama.com
ollama pull llama3.2
```

```python
guard = LLMGuard(
    use_llm=True,
    llm_backend="ollama",
    llm_model="llama3.2",
    llm_base_url="http://localhost:11434",
)
```

**Option 2 — Connect to an existing endpoint (vLLM, LM Studio, LocalAI):**

```python
guard = LLMGuard(
    use_llm=True,
    llm_backend="openai_compatible",
    llm_base_url="http://10.0.0.5:8000",
    llm_model="llama3.1:8b",
)
```

**Option 3 — HuggingFace Transformers (GPU/CPU):**

```python
guard = LLMGuard(
    use_llm=True,
    llm_backend="transformers",
    llm_model="meta-llama/Llama-3.1-8B-Instruct",
    llm_load_in_8bit=True,  # optional: reduce VRAM usage
)
```

> **Note:** HTTP connections to LLM backends (including localhost) log a warning. Use HTTPS in production via a reverse proxy (nginx, Caddy).

#### Ensemble adjudication

By default the three layers run independently and their findings are merged (union). With `llm_adjudicate=True`, the engine instead sends the regex/NER candidates to the LLM, which — in a **single call** — verifies each one (keep / relabel / drop) and adds any PII the other layers missed. Deterministic, checksum-validated regex spans are always kept regardless of the LLM verdict. This sharply reduces NER noise (e.g. job titles mislabeled as names, or a model run on the wrong language):

```python
guard = LLMGuard(
    use_ner=True,
    spacy_model="de_core_news_sm",
    use_llm=True,
    llm_model="gemma3:12b",
    llm_adjudicate=True,   # LLM acts as detector + arbiter in one call
)
```

> Adjudication has no effect in LLM-only deployments (no regex/NER candidates to judge) — the LLM simply runs as a pure detector.

---

## CLI

### Scanning

```bash
# Scan a single text
ai-guard scan --text "TC: 12345678950 card: 4111111111111111"

# Scan from file, JSON output
ai-guard scan --file input.txt --format json

# Disable NER (regex-only, fastest)
ai-guard scan --text "..." --salt "my-salt" --no-ner

# Use Turkish SpaCy model
ai-guard scan --text "..." --model tr_core_news_md

# Batch — each line is scanned independently
ai-guard batch --file lines.txt --format json
```

### SpaCy NER Model Management

```bash
# List all supported SpaCy models
ai-guard spacy list

# Filter by language code
ai-guard spacy list --lang tr
ai-guard spacy list --lang en

# Download a model
ai-guard spacy download en_core_web_sm      # English (default)
ai-guard spacy download tr_core_news_md     # Turkish (recommended)
ai-guard spacy download tr_core_news_lg     # Turkish (higher accuracy, ~318 MB)

# Show installed models
ai-guard spacy installed
```

### On-prem LLM Model Management

```bash
# List available on-prem models
ai-guard models list --recommended

# Download a model via Ollama
ai-guard models pull llama3.1:8b
```

### Example JSON output

```json
{
  "is_clean": false,
  "sanitized_text": "card: [CREDIT_CARD:c5a992a8ea782818]",
  "violations": [
    {
      "entity_type": "CREDIT_CARD",
      "original": "4111111111111111",
      "start": 6,
      "end": 22,
      "action": "hash",
      "replacement": "[CREDIT_CARD:c5a992a8ea782818]"
    }
  ]
}
```

---

## Supported Entity Types

### Regex (built-in, no dependencies)

#### Financial & Identity

| Entity | Default Action | Description |
|---|---|---|
| `CREDIT_CARD` | `hash` | Visa, MC, Amex, Discover — with or without separators; Luhn (mod-10) validated |
| `IBAN` | `hash` | International IBAN — mod-97 checksum validated |
| `SSN` | `hash` | US Social Security Number (123-45-6789) |
| `NIN` | `hash` | UK National Insurance Number (AB123456C) |
| `TC_ID` | `hash` | Turkish national ID — 11 digits, Nüfus İdaresi checksum validated |
| `EU_NATIONAL_ID` | `hash` | Spanish DNI (12345678Z) / NIE (X1234567L), French INSEE (15 digits) |
| `CODICE_FISCALE` | `hash` | Italian tax code (RSSMRA85T10A562S) |
| `VAT_NUMBER` | `warn` | EU VAT (DE/FR/GB/IT/ES/AT/NL prefixed) + Turkish Vergi No (keyword-based) |
| `PASSPORT` | `hash` | Passport numbers — keyword-based (`passport no:`, `pasaport`, `Reisepass`, `passeport`) |
| `DATE_OF_BIRTH` | `hash` | Birth dates — month names (TR/EN/DE/FR) or keyword + numeric/ISO date |
| `VEHICLE_PLATE` | `warn` | Turkish vehicle plates (34 ABC 123) |
| `FINANCIAL_AMOUNT` | `redact` | Monetary amounts (₺/$/€/£, `45.000 TL`, `2.1 milyon TL`) — **off by default** |

#### Contact & Location

| Entity | Default Action | Description |
|---|---|---|
| `EMAIL` | `warn` | RFC-compliant email addresses |
| `PHONE` | `warn` | Turkish (`0`, `+90`), French national (`01 23 45 67 89`), German mobile (`0151 …`), and international E.164 (`+1`, `+44`, …) |
| `ADDRESS` | `warn` | Turkish (Cad., Sok., Mah.), English (Street, Avenue, Road…), French (Rue, Allée…), Spanish (Calle, Avenida…), Italian (Piazza, Corso…), Dutch (straat, gracht…), German (Straße, Weg, Platz…) |
| `POSTAL_CODE` | `warn` | Turkish postal codes (01000–81999) |
| `UK_POSTAL_CODE` | `warn` | British postcodes (SW1A 1AA, GU21 6TH, M1 1AE) |
| `US_ZIP_CODE` | `warn` | US ZIP+4 codes (12345-6789) |

#### Network & Technical

| Entity | Default Action | Description |
|---|---|---|
| `IP_ADDRESS` | `warn` | IPv4 addresses |
| `IPv6` | `warn` | IPv6 addresses (full and compressed forms) |
| `MAC_ADDRESS` | `warn` | Network hardware address (00:1A:2B:3C:4D:5E) |
| `UUID` | `warn` | RFC 4122 UUID / GUID |
| `JWT` | `hash` | JSON Web Token (starts with `eyJ`) |
| `CUSTOM_SECRET` | `hash` | API keys & tokens: OpenAI/Anthropic (`sk-`, `sk-ant-`), Stripe (`sk_live_`), AWS (`AKIA`), Google (`AIza`, `ya29.`), GitHub (`ghp_`), GitLab (`glpat-`), Slack (`xoxb-`, webhook URLs), Twilio (`SK`/`AC`), SendGrid (`SG.`), npm (`npm_`), and PEM private-key blocks |

### SpaCy NER (requires `spacy` + language model)

| Entity | Default Action | Description |
|---|---|---|
| `PERSON` | `hash` | Person names (first + last) — cross-language |
| `ORG` | `warn` | Organization / company names |
| `ADDRESS` | `warn` | Location entities (complements regex) |

> **Language & NER models:** the regex and LLM layers are multilingual by design. The SpaCy layer detects names with the model you load — set `spacy_model` to match your text (`en_core_web_sm`, `de_core_news_sm`, `fr_core_news_sm`, `tr_core_news_md`, …). Running, say, the Turkish model on German text produces noisy results, so pick the model per language (or rely on the LLM layer for cross-language names). A multilingual gazetteer filters out job titles, HR terms, and abbreviations (EN/DE/FR/TR) that NER models commonly mislabel.

### LLM (requires on-prem LLM backend)

| Entity | Default Action | Description |
|---|---|---|
| `PASSPORT` | `hash` | Passport numbers of any country — contextual detection (e.g. `Passport: A12345678`) |
| `EU_NATIONAL_ID` | `hash` | French INSEE, German Personalausweis — contextual variants not caught by regex |
| `UK_POSTAL_CODE` | `warn` | Postcodes in ambiguous contexts |
| `US_ZIP_CODE` | `warn` | ZIP codes in ambiguous contexts |
| `CUSTOM_SECRET` | `hash` | Contextual secrets — `password=VALUE`, `api_key=VALUE`, access codes |
| `SPECIAL_CATEGORY` | `redact` | GDPR Art.9 special-category data — health, religion, ethnicity, political opinion, sexual orientation, trade-union, genetic/biometric. **Off by default** (semantic, LLM-only) |
| *(any above)* | — | LLM supplements and verifies all regex/NER entity types |

> **GDPR special categories:** `SPECIAL_CATEGORY` flags sensitive statements that have no pattern (e.g. "HIV positive", "practising Muslim", "trade-union member") — only the LLM can detect them. It is off by default; enable it under `llm_detector.entities`. Because it is semantic and subjective, expect lower precision than structural entities.

---

## Output Structure

`scan()` and `scan_batch()` return a `ScanResult` per text:

```python
@dataclass
class ScanResult:
    original_text:  str               # unmodified input — contains raw PII
    sanitized_text: str               # anonymized output — safe to forward to LLM
    violations:     List[Violation]   # all detected violations
    is_clean:       bool              # True if no violations found

    def redacted(self) -> dict:       # PII-free dict — safe for logging and APIs
        ...

@dataclass
class Violation:
    entity_type: str          # e.g. "CREDIT_CARD", "EMAIL"
    original:    str          # raw value from input — contains PII
    start:       int          # start index in original text
    end:         int          # end index in original text
    action:      Action       # WARN | HASH | REDACT | MASK
    replacement: str | None   # "[TYPE:16hex]" for hash, "[TYPE]" for redact, masked value for mask, None for warn
    confidence:  float        # 1.0 regex/checksum · 0.85 NER/LLM
```

---

## Security

### Hashing

Sensitive values are replaced in the sanitized text using the format `[TYPE:16hex]` (64-bit entropy):

```
4532 0151 1283 0366  →  [CREDIT_CARD:ea782818c5a992a8]
12345678950          →  [TC_ID:86349f34a1bc2d5e]
```

### Salt

Salt prevents rainbow table attacks. Set it via environment variable in production:

```python
import os
guard = LLMGuard(salt=os.environ["LLMGUARD_SALT"])
```

> Never hardcode the salt in source code or config files.

### Safe Logging

`ScanResult.original_text` and `violations[].original` contain raw PII. Use `result.redacted()` for logs and API responses:

```python
result = guard.scan(text)

# ✗ Leaks PII to logs
logger.info("result: %s", result.original_text)

# ✓ Safe — no raw PII
logger.info("result: %s", result.redacted())
```

### Input Size Limit

Inputs exceeding **500 KB** raise a `ValueError`. Split large documents into smaller chunks before scanning.

---

## Environment Variables

| Variable | Description |
|---|---|
| `LLMGUARD_SALT` | Hash salt |
| `LLMGUARD_LLM_URL` | LLM service base URL |
| `LLMGUARD_LLM_MODEL` | LLM model name |
| `LLMGUARD_LLM_API_KEY` | API key for OpenAI-compatible backends |
| `LLMGUARD_LLM_TIMEOUT` | LLM request timeout in seconds |
| `LLMGUARD_SPACY_MODEL` | SpaCy model name |

---

## Project Structure

```
ai-guard/
├── src/ai_guard/
│   ├── guard.py              # LLMGuard — main interface
│   ├── __main__.py           # CLI entry point
│   ├── core/
│   │   ├── engine.py         # DetectionEngine — overlap resolution, action application
│   │   └── models.py         # Action, Violation, ScanResult
│   ├── detectors/
│   │   ├── base.py           # BaseDetector ABC
│   │   ├── regex_detector.py # 25+ regex patterns with checksum/Luhn validation
│   │   ├── ner_detector.py   # SpaCy NER (multilingual) + gazetteer FP filter
│   │   └── llm_detector.py   # LLM-based detection with hallucination filter
│   ├── llm/
│   │   ├── backends/         # ollama, openai_compat, transformers
│   │   ├── model_catalog.py  # Supported model list
│   │   └── prompt.py         # PII detection prompt builder
│   ├── config/
│   │   └── loader.py         # YAML loader, env var overrides
│   └── utils/
│       └── hashing.py        # SHA-256 + salt
├── tests/
│   ├── unit/                 # Component-level tests
│   └── integration/          # Scenario and adversarial tests
├── config/
│   └── default.yaml          # Example policy file
└── pyproject.toml
```

---

## Testing

```bash
# All tests
uv run pytest tests/unit/ tests/integration/

# Unit tests only
uv run pytest tests/unit/

# NER tests (requires SpaCy model)
uv run pytest -m ner

# Live LLM tests against a real Ollama model (auto-skipped if unavailable)
uv run pytest -m slow tests/integration/test_llm_live.py
# Pick the model: AIGUARD_TEST_LLM_MODEL=llama3.2:1b uv run pytest -m slow ...

# Skip slow tests (fast run)
uv run pytest -m "not slow"

# Coverage report
uv run pytest --cov=src/ai_guard --cov-report=term-missing
```

---

## Known Limitations

| Case | Description |
|---|---|
| `4111  1111  1111  1111` | Double-space separator bypasses card detection |
| `4111.1111.1111.1111` | Dot separator not supported |
| Adjacent IBANs | Two IBANs without separator cannot be parsed separately |
| Cyrillic homoglyphs | `аli@test.com` (Cyrillic `а`) bypasses ASCII regex |
| Transformers backend | Not tested with a real model — mock-only unit tests |
| `US_ZIP_CODE` | Plain 5-digit ZIPs only matched with a `ZIP:` keyword; otherwise ZIP+4 (`12345-6789`) required to avoid false positives |
| `EU_NATIONAL_ID` | Spanish DNI/NIE and French INSEE via regex; German IDs need the LLM layer |
| `PASSPORT` | Regex requires a passport keyword (`passport no:`, `pasaport`, `Reisepass`, `passeport`); the LLM layer catches unlabeled cases |
| `FINANCIAL_AMOUNT` / `VAT_NUMBER` | `FINANCIAL_AMOUNT` is off by default (enable for confidential docs); bare Turkish Vergi No needs a keyword |
| Multilingual NER | One SpaCy model loads per instance — set `spacy_model` to match the text language; auto language detection is not yet built in |
| European addresses | Regex requires a recognizable street-type keyword (Straße, Rue, Calle…); unnumbered informal addresses may be missed |
| Turkish NER (`tr_core_news_trf`) | Transformer model incompatible with SpaCy 3.5+ — use `tr_core_news_md` or `tr_core_news_lg` |
| Turkish NER quality | `tr_core_news_md/lg` trained on news text; may miss names in non-standard contexts. For best results combine with `use_llm=True` |

---

## Development

```bash
uv sync --dev
uv run pytest tests/unit/ tests/integration/
```

Requires Python 3.11+.

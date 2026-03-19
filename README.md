# ai-guard

**PII detection and anonymization for LLM inputs** — hybrid regex + NER + on-prem LLM engine.

`ai-guard` scans text for personally identifiable information (PII) before it reaches an LLM, and either warns about or replaces the sensitive data with salted SHA-256 hashes. It supports Turkish and English out of the box.

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
# Name: Ali Veli, card: [CREDIT_CARD:ea782818], email: ali@example.com
```

---

## Features

- **Hybrid detection** — Regex + SpaCy NER + on-prem LLM (Ollama, OpenAI-compatible, HuggingFace Transformers)
- **Two actions** — `warn` (keep text, report violation) and `hash` (replace with `[TYPE:8hex]` using SHA-256 + salt)
- **Rainbow table protection** — user-defined salt for all hashes
- **Two APIs** — method chaining (programmatic) and YAML (declarative)
- **CLI** — `ai-guard scan`, `ai-guard batch`, `ai-guard models`
- **Turkish support** — TC identity number, IBAN, postal codes, Turkish address patterns, Turkish SpaCy model

---

## Installation

Clone and install with [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/oguzhantopcu0/ai-guard.git
cd ai-guard
uv sync
```

To use SpaCy NER (PERSON, ORG, ADDRESS detection):

```bash
# English model
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# Turkish model (optional)
# uv pip install <tr_core_news_sm wheel URL>
```

SpaCy is not required if you only need regex-based detection.

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
  Customer: Ali Veli, TC: 12345678901
  Card: 4532 0151 1283 0366
  Email: ali.veli@example.com
""")

print(result.sanitized_text)
# Customer: Ali Veli, TC: [TC_ID:86349f34]
# Card: [CREDIT_CARD:ea782818]
# Email: ali.veli@example.com   ← warn: text is kept

for v in result.violations:
    print(f"[{v.action.value}] {v.entity_type}: {v.original!r}")
# [hash] TC_ID: '12345678901'
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

### On-prem LLM (Ollama)

```python
guard = LLMGuard(
    use_llm=True,
    llm_model="llama3.1:8b",
    llm_base_url="http://localhost:11434",
)
```

### HuggingFace Transformers (GPU/CPU)

```python
guard = LLMGuard(
    use_llm=True,
    llm_backend="transformers",
    llm_model="meta-llama/Llama-3.1-8B-Instruct",
    llm_load_in_8bit=True,  # optional: reduce VRAM usage
)
```

---

## CLI

```bash
# Scan a single text
ai-guard scan --text "TC: 12345678901 card: 4111111111111111"

# Scan from file, JSON output
ai-guard scan --file input.txt --format json

# Disable NER
ai-guard scan --text "..." --salt "my-salt" --no-ner

# Turkish SpaCy model
ai-guard scan --text "..." --model tr_core_news_sm

# Batch — each line is scanned independently
ai-guard batch --file lines.txt --format json

# List available on-prem models
ai-guard models list --recommended

# Download a model via Ollama
ai-guard models pull llama3.1:8b
```

### Example JSON output

```json
{
  "is_clean": false,
  "sanitized_text": "card: [CREDIT_CARD:c5a992a8]",
  "violations": [
    {
      "entity_type": "CREDIT_CARD",
      "original": "4111111111111111",
      "start": 6,
      "end": 22,
      "action": "hash",
      "replacement": "[CREDIT_CARD:c5a992a8]"
    }
  ]
}
```

---

## Supported Entity Types

### Regex (built-in, no dependencies)

| Entity | Default Action | Description |
|---|---|---|
| `CREDIT_CARD` | `hash` | Visa, MC, Amex, Discover — with or without separators |
| `EMAIL` | `warn` | RFC-compliant email addresses |
| `PHONE` | `warn` | Turkish phone numbers (`0`, `+90`, `90` prefix) |
| `IBAN` | `hash` | International IBAN (case-insensitive) |
| `IP_ADDRESS` | `warn` | IPv4 addresses |
| `IPv6` | `warn` | IPv6 addresses (full and compressed forms) |
| `TC_ID` | `hash` | Turkish national ID — 11 digits |
| `ADDRESS` | `warn` | Turkish address patterns (Cad., Sok., Mah., Blv.) |
| `POSTAL_CODE` | `warn` | Turkish postal codes (01000–81999) |
| `UUID` | `warn` | RFC 4122 UUID / GUID |
| `SSN` | `hash` | US Social Security Number (123-45-6789) |
| `MAC_ADDRESS` | `warn` | Network hardware address (00:1A:2B:3C:4D:5E) |
| `JWT` | `hash` | JSON Web Token (starts with `eyJ`) |
| `NIN` | `hash` | UK National Insurance Number (AB123456C) |

### SpaCy NER (requires `spacy` + language model)

| Entity | Default Action | Description |
|---|---|---|
| `PERSON` | `hash` | Person names |
| `ORG` | `warn` | Organization names |
| `ADDRESS` | `warn` | Location/address entities (overlaps with regex) |

### LLM (requires on-prem LLM backend)

| Entity | Default Action | Description |
|---|---|---|
| `CUSTOM_SECRET` | `hash` | Contextual secrets — passwords, API keys, tokens |
| *(any above)* | — | LLM can also verify/supplement all regex/NER types |

---

## Output Structure

`scan()` and `scan_batch()` return a `ScanResult` per text:

```python
@dataclass
class ScanResult:
    original_text:  str               # unmodified input
    sanitized_text: str               # anonymized output
    violations:     List[Violation]   # all detected violations
    is_clean:       bool              # True if no violations found

@dataclass
class Violation:
    entity_type: str          # e.g. "CREDIT_CARD", "EMAIL"
    original:    str          # raw value from input
    start:       int          # start index in original text
    end:         int          # end index in original text
    action:      Action       # Action.WARN | Action.HASH
    replacement: str | None   # "[TYPE:8hex]" for hash, None for warn
```

---

## Security

### Hashing

Sensitive values are replaced in the sanitized text using the format `[TYPE:8hex]`:

```
4532 0151 1283 0366  →  [CREDIT_CARD:ea782818]
12345678901          →  [TC_ID:86349f34]
```

### Salt

Salt prevents rainbow table attacks. Set it via environment variable in production:

```python
import os
guard = LLMGuard(salt=os.environ["LLMGUARD_SALT"])
```

> Never hardcode the salt in source code or config files.

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
│   │   ├── regex_detector.py
│   │   ├── ner_detector.py   # SpaCy NER (English + Turkish)
│   │   └── llm_detector.py   # LLM-based detection
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
uv run pytest

# Unit tests only
uv run pytest tests/unit/

# NER tests (requires SpaCy model)
uv run pytest -m ner

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

---

## Development

```bash
uv sync --dev
uv run pytest
```

Requires Python 3.11+.

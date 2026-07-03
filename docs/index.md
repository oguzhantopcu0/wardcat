# wardcat

**PII detection and anonymization for LLM inputs** — a hybrid engine that scans
text for personally identifiable information (PII) *before* it reaches an LLM,
and either warns about or replaces the sensitive data with salted SHA-256
hashes.

Four detection layers cooperate behind one interface:

| Layer | What it does | Cost |
|---|---|---|
| **regex** | Deterministic structural PII — email, card (Luhn), IBAN (mod-97), TC ID, secrets… | free, exhaustive |
| **ner** | SpaCy Named Entity Recognition — names, orgs, locations | fast, per-language model |
| **gliner** | Zero-shot transformer NER (GLiNER2) — names + rich PII, one multilingual model | medium |
| **llm** | On-prem LLM — contextual/semantic PII (GDPR Art.9, contextual secrets) | slow, strongest context |

**Detection is opt-in:** a bare `Wardcat()` detects nothing — you enable the
entities you want.

```python
import os
from wardcat import Wardcat, Entity, Action

guard = (
    Wardcat(salt=os.environ.get("WARDCAT_SALT", ""))
    .add_entity(Entity.CREDIT_CARD, Action.HASH)
    .add_entity(Entity.EMAIL,       Action.WARN)
    .add_entity(Entity.TC_ID,       Action.HASH)
)

result = guard.scan("Name: Ali Veli, card: 4532 0151 1283 0366, email: ali@example.com")
print(result.sanitized_text)
# Name: Ali Veli, card: [CREDIT_CARD:ea782818c5a992a8], email: ali@example.com
```

## Highlights

- **Hybrid detection** across four cooperating layers, merged with a
  confidence-first overlap resolver (a deterministic regex span always wins).
- **Checksum validation** — TC_ID, IBAN, and CREDIT_CARD are mathematically
  verified before flagging, eliminating false positives.
- **Four actions** — `warn`, `hash` (salted SHA-256), `redact`, `mask` — all
  pluggable via a registry.
- **Value propagation** — once any layer detects a value, every occurrence of it
  can be anonymized ([Configuration](guide/configuration.md)).
- **Degraded-scan visibility** — if a layer can't run (e.g. LLM backend down),
  it's recorded on `ScanResult.warnings` instead of failing silently.
- **Safe logging** — `result.redacted()` returns a PII-free dict.

## Where next

- [Installation](installation.md) — base install and optional extras.
- [Quickstart](quickstart.md) — the programmatic and YAML APIs.
- [Detection layers](guide/layers.md) — regex, NER, GLiNER, and the LLM layer.
- [API reference](reference/aiguard.md) — generated from the source docstrings.

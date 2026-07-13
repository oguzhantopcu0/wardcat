# wardcat

**PII detection and anonymization for LLM inputs** — a hybrid engine that scans
text for personally identifiable information (PII) *before* it reaches an LLM,
and either warns about or replaces the sensitive data with salted SHA-256
hashes.

Three detection layers cooperate behind one interface:

| Layer | What it does | Cost |
|---|---|---|
| **regex** | Deterministic structural PII — email, card (Luhn), IBAN (mod-97), TC ID, secrets… | free, exhaustive |
| **ner** | SpaCy Named Entity Recognition — names, orgs, locations | fast, per-language model |
| **llm** | On-prem LLM — contextual/semantic PII (GDPR Art.9, contextual secrets) | slow, strongest context |

**Detection is opt-in:** a bare `Wardcat()` detects nothing — you configure the
layers and entities you want. The full picture — all three layers, a per-entity
action policy, the semantic guardrail, and anonymization:

```python
import os
from wardcat import Wardcat, Backend, Entity, Action

# One guard, all three detection layers — regex + SpaCy NER + on-prem LLM.
guard = (
    Wardcat(salt=os.environ["WARDCAT_SALT"])               # your app supplies the salt
    .with_ner(language="tr")                               # names / orgs  (needs wardcat[ner])
    .with_llm(backend=Backend.OLLAMA, model="gemma3:12b")  # contextual + semantic (needs Ollama)
    # An action per entity: hash IDs, mask contact details, redact names, flag IPs.
    .add_entities([Entity.CREDIT_CARD, Entity.IBAN, Entity.TC_ID], action=Action.HASH)
    .add_entities([Entity.EMAIL, Entity.PHONE], action=Action.MASK)
    .add_entity(Entity.PERSON, Action.REDACT)
    .add_entity(Entity.IP_ADDRESS, Action.WARN)
)

text = "Ben Ahmet Yılmaz, kartım 4111 1111 1111 1111, e-posta ahmet@example.com."

# 1) Semantic guardrail: does the text contain sensitive info at all? (holistic LLM yes/no)
if guard.is_sensitive(text):

    # 2) Anonymize the PII before the text is stored, logged, or forwarded.
    result = guard.scan(text)
    print(result.sanitized_text)
    # Ben [PERSON], kartım [CREDIT_CARD:b22b36262d8d2769], e-posta a****@example.com.

    print(result.redacted())   # PII-free dict, safe for logs / APIs
```

!!! tip "The NER and LLM layers are optional"
    Regex-only detection needs no models — just
    `Wardcat(salt=...).add_entity(Entity.CREDIT_CARD, Action.HASH)` and `.scan(...)`.
    Add `with_ner(...)` / `with_llm(...)` when you want names or contextual/semantic
    detection.

## Highlights

- **Hybrid detection** across three cooperating layers, merged with a
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
- [Detection layers](guide/layers.md) — regex, NER, and the LLM layer.
- [API reference](reference/wardcat.md) — generated from the source docstrings.

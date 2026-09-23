# Quickstart

## Programmatic API

Use the `Entity` and `Action` constants — your IDE autocompletes them and a typo
is caught at edit time. They are interchangeable with the string forms
(`Entity.EMAIL == "EMAIL"`).

```python
from wardcat import Wardcat, Entity, Action

guard = (
    Wardcat(salt="my-secret-salt")
    .add_entity(Entity.CREDIT_CARD, Action.HASH)
    .add_entity(Entity.EMAIL,       Action.WARN)
    .add_entity(Entity.TC_ID,       Action.HASH)
)

result = guard.scan("Customer: Ali Veli, TC: 12345678950, card: 4111 1111 1111 1111")

print(result.sanitized_text)
for v in result.violations:
    print(f"[{v.action}] {v.entity_type}: {v.original!r}")
```

Enable many at once with `add_entities()` (a list, a `{name: action}` mapping, or
a `{name: {action, layers}}` mapping), or turn on **everything** then prune:

```python
from wardcat import Wardcat, Entity

guard = (
    Wardcat(salt="s")
    .add_entity(Entity.ALL, action="hash")   # everything on, hashed
    .remove_entity(Entity.ORG)               # …except organisation names
)
```

Predefined groups pair naturally with `add_entities()`:

```python
from wardcat import Wardcat, turkish_entities

guard = Wardcat(salt="s")
guard.add_entities(turkish_entities(), action="hash")          # a whole group, one action
guard.add_entities(["EMAIL", "CREDIT_CARD", "IBAN"], action="redact")
guard.add_entities({                                           # per-entity actions and layers
    "CREDIT_CARD":      "hash",
    "EMAIL":            {"action": "mask"},
    "SPECIAL_CATEGORY": {"action": "redact", "layers": ["llm"]},
})
```

The groups — `core_entities`, `financial_entities`, `turkish_entities`,
`european_entities`, `uk_entities`, `us_entities`, `network_entities`,
`identity_entities`, `all_entities` — are importable from `wardcat`. Every
entity, with its default action, is on the [entity types](reference/entities.md)
page.

## Declarative API (YAML)

```python
from wardcat import Wardcat
guard = Wardcat(config_path="config/my_policy.yaml")
```

```yaml
salt: ""          # read from env in production
entities:
  CREDIT_CARD: { enabled: true, action: hash }
  EMAIL:       { enabled: true, action: warn }
  TC_ID:       { enabled: true, action: hash }
```

## The result

`scan()` returns a [`ScanResult`](reference/models.md#wardcat.ScanResult):

```python
result = guard.scan(text)
result.sanitized_text   # anonymized output — safe to forward to an LLM
result.violations       # list[Violation] — each with entity_type, original, action, confidence
result.is_clean         # True if nothing was found
result.warnings         # non-empty if a layer could not run (degraded scan)
result.redacted()       # PII-free dict — safe for logs / API responses
```

!!! warning "Raw PII"
    `result.original_text` and `violations[].original` contain raw PII. Use
    `result.redacted()` for logs and API responses.

## Batch

```python
results = guard.scan_batch(["ali@example.com", "Card: 4111 1111 1111 1111", "Clean."])
for r in results:
    print(r.is_clean, len(r.violations))
```

## Examples

Runnable scripts in [`examples/`](https://github.com/oguzhantopcu0/wardcat/tree/main/examples):

| File | Shows |
|---|---|
| `demo.py` | Programmatic + YAML APIs |
| `batch_and_async.py` | `scan_batch` and the async API (regex-only, no services) |
| `llm_hybrid.py` | regex + NER + LLM with ensemble adjudication (needs Ollama) |
| `all_layers.py` | all three layers on one guard, with adjudication (needs Ollama + a SpaCy model) |
| `reversible_roundtrip.py` | `Action.TOKENIZE` out, `restore()` back — regex-only, stubbed model, runs offline |
| `asgi_middleware.py` | Copy-paste ASGI middleware (FastAPI/Starlette) that scans request bodies — wardcat ships no web-framework code; this is a self-contained example |

Next: enable the [detection layers](guide/layers.md) you need.

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

result = guard.scan("Customer: Ali Veli, TC: 12345678950, card: 4532 0151 1283 0366")

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

Next: enable the [detection layers](guide/layers.md) you need.

# Configuration & policy

Configuration is **explicit** — pass constructor arguments or a YAML
`config_path`. The library never reads environment variables; read any secrets in
*your* application and hand them to the constructor.

## Entity policy (write API)

| Method | Effect |
|---|---|
| `add_entity(entity, action, layers=None)` | Enable one entity (or `Entity.ALL`) |
| `add_entities(mapping_or_list, ...)` | Enable many in one rebuild |
| `remove_entity(entity)` / `remove_entities([...])` | Disable |
| `change_entity_action(entity, action)` | Retarget an **already-enabled** entity's action |

Read API: `enabled_entities()`, `get_entity_action(entity)`, `entity_policy()`,
and the static `Wardcat.supported_entities(layer=None)`.

## Actions

`warn` (keep text, report only) · `hash` (`[TYPE:16hex]`, salted SHA-256) ·
`redact` (`[TYPE]`) · `mask` (entity-aware partial masking). When `action` is
omitted it defaults to `hash` (with a one-time warning). Actions are
[pluggable](extending.md#custom-actions).

## Value propagation

Model-based layers sometimes report a repeated value only once. `with_propagation()`
anonymizes **every** whole-token occurrence once any layer detects a value:

```python
guard = Wardcat(salt="s").with_gliner().add_entity("PERSON").with_propagation()
```

Off by default (it can over-redact); only exact, token-bounded matches at least
`min_length` chars (default 3) propagate, and deterministic regex spans still win
overlaps.

## Allowlist / denylist

```python
guard.add_allowlist(["no-reply@company.com"])                 # never flag
guard.add_denylist([{"value": "ProjectX", "entity_type": "CUSTOM_SECRET"}])  # always flag
```

## Degraded scans

If a layer cannot run (most commonly the LLM backend being unreachable), the scan
still returns the other layers' results and records the failure:

```python
res = guard.scan(text)
if res.warnings:
    logger.warning("PII scan degraded: %s", res.warnings)
```

## YAML reference

```yaml
salt: ""
use_ner: false
propagate_matches: false
propagate_min_length: 3

entities:
  CREDIT_CARD: { enabled: true, action: hash }
  EMAIL:       { enabled: true, action: warn }

gliner_detector:
  enabled: false
  model: "fastino/gliner2-privacy-filter-PII-multi"
  threshold: 0.5
  chunk_size: 1500

llm_detector:
  enabled: false
  backend: ollama
  model: llama3.2
  adjudicate: false
```

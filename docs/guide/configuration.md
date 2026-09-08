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
`redact` (`[TYPE]`) · `mask` (entity-aware partial masking) · `tokenize`
(`[TYPE_1]`, [reversible](reversible.md)). When `action` is omitted it defaults
to `hash` (with a one-time warning). Actions are
[pluggable](extending.md#custom-actions).

## The LLM layer's own entity policy

`with_ner()` enables no entity by itself — you opt in with `add_entity`. **`with_llm()`
does not work that way**: the LLM layer carries its own default policy of **31 entity
types, 27 of them switched on** (`ORG`, `LOCATION`, `NRP` and `SPECIAL_CATEGORY` ship
off), each with its own action, so

```python
guard = Wardcat(salt="s").with_llm(...).add_entity(Entity.EMAIL, Action.TOKENIZE)
guard.enabled_entities()                 # 27 types, not 1
guard.get_entity_action(Entity.PERSON)   # 'hash' — nobody asked for this
```

detects and anonymizes much more than the one type named, under the policy's actions
rather than the one just configured. The first scan logs a one-time warning listing
what came along. To take control:

```python
guard.add_entity(Entity.PERSON, Action.REDACT, layers=["llm"])  # override one
guard.remove_entity(Entity.PERSON)                              # drop one
guard.remove_entity(Entity.ALL)                                 # start from nothing
Wardcat(config_path="policy.yaml")                              # replace it wholesale
```

## Phone regions

`PHONE` is matched by a precision-first pattern covering TR/FR/DE national formats
plus E.164. For national formats elsewhere, name the regions you serve and
detection moves to libphonenumber:

```python
guard.with_phone_regions("GB", "ES", "US")   # CLDR codes
guard.with_phone_regions()                   # back to the built-in pattern
```

Needs `pip install "wardcat[phone]"`; without it the pattern is used and a warning
is logged. Matches report `0.90` confidence rather than `0.97` — a numbering-plan
check is weaker than a checksum, and each extra region widens what counts as a
number, so add the regions you serve rather than all of them.

## Confidence floor

Every detection carries a confidence, tiered by how strong the evidence is:

| Tier | Value | What earns it |
|---|---|---|
| checksum | `1.00` | a card passing Luhn, an IBAN passing mod-97, a Bitcoin address |
| structural | `0.97` | a distinctive format — email, JWT, IPv4 |
| fuzzy | `0.90` | a keyword heuristic — a street address, a cued password or handle |
| model | `0.85` | a span from the NER or LLM layer |
| uncued | `0.70` | a bare digit run passing a checksum whose own odds are weak |

`min_confidence` is the floor below which a span is dropped before any action is
applied. It defaults to `0.8`, which sits above that last tier and below every
other one, so those uncued matches are found but left alone:

```python
guard.with_min_confidence(0.6)    # act on uncued matches too
guard.with_min_confidence(0.95)   # checksummed and structural only
```

The floor is applied **after** overlap resolution, so a stronger span still wins
its overlap first: a phone number that also satisfies the NHS checksum resolves
as `PHONE`, rather than being dropped as a weak NHS match.

## Value propagation

Model-based layers sometimes report a repeated value only once. `with_propagation()`
anonymizes **every** whole-token occurrence once any layer detects a value:

```python
guard = Wardcat(salt="s").with_ner(language="en").add_entity("PERSON").with_propagation()
```

Off by default (it can over-redact); only exact, token-bounded matches at least
`min_length` chars (default 3) propagate, and deterministic regex spans still win
overlaps.

## Allowlist / denylist

```python
guard.add_allowlist(["no-reply@example.com"])                 # never flag
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
min_confidence: 0.8      # drop spans scoring below this
phone_regions: []        # CLDR codes for libphonenumber-backed PHONE
propagate_matches: false
propagate_min_length: 3

entities:
  CREDIT_CARD: { enabled: true, action: hash }
  EMAIL:       { enabled: true, action: warn }

llm_detector:
  enabled: false
  backend: ollama
  model: llama3.2
  adjudicate: false
```

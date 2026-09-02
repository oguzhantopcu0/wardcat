# Reversible masking

`hash`, `redact` and `mask` are one-way: once the text leaves the anonymizer the
value is gone from it for good. `Action.TOKENIZE` is the reversible one — each
value becomes a numbered placeholder and the mapping stays on the `ScanResult`,
in memory, on your side of the wire.

That is the shape of an LLM round trip: mask on the way out, restore on the way
back, with the model never seeing a real value.

## Masking

```python
from wardcat import Wardcat, Entity, Action

guard = (
    Wardcat(salt="s")
    .add_entities([Entity.EMAIL, Entity.CREDIT_CARD], action=Action.TOKENIZE)
)

result = guard.scan("Mail ali@example.com about card 4532 0151 1283 0366")
result.sanitized_text
# 'Mail [EMAIL_1_9f3a2c8b71d4] about card [CREDIT_CARD_1_9f3a2c8b71d4]'
```

A placeholder is **`[TYPE_index_contextid]`**. The index is per entity type in
order of appearance, and the same value always gets the same token — a name that
repeats stays one referent, so the model can still reason about it:

```python
guard.scan("Ali wrote to Veli; reply to Ali").sanitized_text
# '[PERSON_1_4c81…] wrote to [PERSON_2_4c81…]; reply to [PERSON_1_4c81…]'
```

The index restarts at 1 for every scan, so it stays short; the **context id** is
drawn once per scan (`result.context_id`) and shared by that scan's tokens. See
[Why the context id](#why-the-context-id) — it is the difference between a
misrouted answer restoring the wrong person's data and restoring nothing at all.

Which layer found the value does not matter: anonymization runs after detection,
so a span from regex, SpaCy NER or the LLM layer is tokenized and restored the
same way. Only the confidence in the source list gives the layer away — `1.00`
for a checksum-validated regex match, `0.85` for a model-based one.

## Restoring

```python
answer = call_llm(result.sanitized_text)
print(result.restore(answer))
```

```text
I have emailed ali@example.com about the charge on 4532 0151 1283 0366.

--- Sources ---
[1] [EMAIL_1_9f3a2c8b71d4] → ali@example.com (EMAIL · tokenize · confidence 0.97)
[2] [CREDIT_CARD_1_9f3a2c8b71d4] → 4532 0151 1283 0366 (CREDIT_CARD · tokenize · confidence 1.00)
```

Printing a `RestoredText` appends that source list: each placeholder that was put
back, in the order it appears in the text, with the filter that caught it, the
action applied, the value it stood for, an `xN` count when it occurs more than
once, and the detection confidence.

| | |
|---|---|
| `.text` | the restored text alone |
| `.sources_block()` | the list alone — `title=` renames it, `notes=False` drops the trailing summary |
| `.with_sources()` | text + list; what `str(restored)` returns |
| `.substitutions` | the same information as `Substitution` records |
| `.unrestored` | detected values that were *not* put back, and why |
| `.is_complete` | `False` if a placeholder was left sitting in the text (ambiguous or foreign) |

`restore()` with no argument reverses `sanitized_text` itself, which round-trips
back to the original input — a cheap way to assert a policy is reversible in
tests.

## Why the context id

Two requests arriving together both hold an "`EMAIL` number 1". If placeholders
were just `[EMAIL_1]`, the two scans would produce the *same string*, and
restoring one request's answer against the other's result would match it and
substitute the wrong person's address — silently, with no error and nothing in
`unrestored`. That is the worst outcome this library can produce.

The context id makes those tokens distinct, so the mistake has nothing to match:

```python
mine   = guard.scan("Ali: ali@example.com")      # [EMAIL_1_9f3a2c8b71d4]
theirs = guard.scan("Beyza: beyza@example.com")  # [EMAIL_1_4c81b70a2e55]

theirs.restore("Mailed [EMAIL_1_9f3a2c8b71d4].")
# text        unchanged — nothing was substituted
# unrestored  [UnrestoredValue(entity_type='EMAIL', reason='foreign', …)]
# is_complete False
```

The id is 12 hex characters — 48 bits, so with a hundred thousand results alive
at once the chance any two collide is about 1.8e-5, and request-scoped use puts
it near 1e-9. It is drawn per scan, needs no shared counter, and holds across
processes.

Two knobs:

```python
result.restore(answer, strict=True)          # raise ContextMismatch instead of reporting
turn2.restore(answer, also=[turn1])          # an earlier turn's tokens are legitimate here
```

`strict=True` is the right default for a request handler. `also=` is for
multi-turn exchanges, where an answer may carry placeholders from a previous
scan in the same conversation — those are restored too, and anything else still
raises.

A mangled token is covered by the same mechanism: if the model returns
`[EMAIL_1_9f3a2c8b71d5]` for `[EMAIL_1_9f3a2c8b71d4]`, there is no match, so the
value simply does not come back. The model's accuracy decides *how much* is
restored; it can never decide *whether what comes back is correct*.

## Prompting the model to keep placeholders

Restoring needs the model to reproduce the placeholders verbatim. Whether it does
is a property of the model — and, more than anything, of the instruction you give
it. Measured on a local Qwen2.5-1.5B with the same text and three runs each:

| instruction | placeholders preserved |
|---|---|
| "Copy every placeholder in square brackets EXACTLY as written." | 0 / 9 |
| "The text contains placeholders like `[EMAIL_1]`. Copy every placeholder EXACTLY as written, character for character. Never invent placeholders." | 6 / 9 |

One model, three runs per instruction, nine placeholder slots each: the direction
is worth acting on, the exact rate is not. A starting point:

```python
INSTRUCTION = (
    "The text contains placeholders like [EMAIL_1_9f3a2c8b71d4]. "
    "Copy every placeholder EXACTLY as written, character for character. "
    "Never invent placeholders and never renumber them."
)
answer = call_llm(INSTRUCTION + "\n\n" + payload.sanitized_text)
```

Two things this measurement settled:

**The context id does not cost recall.** The same comparison with the id stripped
out preserved *fewer* placeholders, not more — consistent with the failure mode
seen elsewhere, where a model turns `[CREDIT_CARD_1]` into `[CREDIT_CARD_2]` but
copies `[CREDIT_CARD_49]` untouched. A trailing hex suffix reads as an opaque
identifier to copy rather than an item to renumber.

**A model too weak to copy placeholders is not a safety problem.** Small models
paraphrase them away or drop them entirely; Qwen2.5-0.5B and SmolLM2-135M both
did in testing. Nothing wrong is substituted — the value simply does not come
back, and it is reported in `.unrestored`. Check it, and decide what your
application does about a partial restore:

```python
restored = payload.restore(answer)
if restored.unrestored:
    log.warning("partial restore: %s", [(u.entity_type, u.reason) for u in restored.unrestored])
```

## Which actions can be reversed

| Action | Reversible | Why |
|---|---|---|
| `tokenize` | always | one placeholder per distinct value, by construction |
| `hash` | in practice | a salted digest differs per distinct value |
| `redact` | only if unique | every value of a type collapses onto `[TYPE]` |
| `mask` | only if unique | two values can mask to the same string |
| `warn` | nothing to do | the original text was never replaced |

A placeholder that stood for more than one value is **not** guessed: it is left
in the text and reported in `.unrestored` with reason `ambiguous` (and called out
under the source list). Scanned with a one-way action and want a reversible view
without re-detecting? `result.reapply(Action.TOKENIZE)` reuses the spans and
re-runs only the anonymization stage.

## Keep the payload uniform

A guard usually mixes actions, and `with_llm()` in particular switches on its own
entity policy — one whose defaults include `warn`. **`warn` reports a value without
replacing it**, so a scan you believe is masked can hand the model a phone number
in the clear:

```text
[PERSON:5687e6a708553da8], [EMAIL_1_9f3a…], tel +90 532 123 45 67
  PERSON  hash        EMAIL  tokenize        PHONE  warn  ← still in the text
```

Deriving a uniformly tokenized payload closes that, and makes every token short and
alike while it is at it. Detection is not repeated, only the cheap anonymization
stage:

```python
result  = guard.scan(text)
payload = result.reapply(Action.TOKENIZE)   # every span becomes [TYPE_N]

answer = call_llm(payload.sanitized_text)
print(payload.restore(answer))              # restore through the same object
```

```text
mixed   : [PERSON:5687e6a708553da8], [EMAIL_1_9f3a…], tel +90 532 123 45 67
uniform : [PERSON_1_0c29…], [EMAIL_1_0c29…], tel [PHONE_1_0c29…]
```

Restore through the object you sent — `payload`, not `result`: the placeholders in
the answer are the ones `reapply` produced.

## Handing the map to another process

`restore()` needs the `ScanResult`. When the answer comes back somewhere else —
a queue worker, another request — carry the map yourself:

```python
vault = result.token_map     # {'[EMAIL_1_9f3a2c8b71d4]': 'ali@example.com'}
context = result.context_id  # carry it too, to spot a token from another scan
```

!!! danger "The reverse map is raw PII"
    `token_map`, `restore()` and the source list all carry the original values by
    design — that is what reversibility means. Keep them on the trusted side, out
    of logs and telemetry ([`redacted()`](../reference/models.md) stays the safe
    view), and prefer a one-way action when you do not need the values back.
    Placeholder numbering is per scan, so a `ScanResult` is the only thing that
    can reverse its own text.

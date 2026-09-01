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
# 'Mail [EMAIL_1] about card [CREDIT_CARD_1]'
```

Placeholders are numbered **per entity type in order of appearance**, and the
same value always gets the same token — a name that repeats stays one referent,
so the model can still reason about it:

```python
guard.scan("Ali wrote to Veli; reply to Ali").sanitized_text
# '[PERSON_1] wrote to [PERSON_2]; reply to [PERSON_1]'
```

Numbering restarts at 1 for every scan, and concurrent scans (`scan_batch`,
`scan_async`) never share a counter.

## Restoring

```python
answer = call_llm(result.sanitized_text)
print(result.restore(answer))
```

```text
I have emailed ali@example.com about the charge on 4532 0151 1283 0366.

--- Sources ---
[1] [EMAIL_1] → ali@example.com (EMAIL · tokenize · confidence 0.97)
[2] [CREDIT_CARD_1] → 4532 0151 1283 0366 (CREDIT_CARD · tokenize · confidence 1.00)
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
| `.is_complete` | `False` if a placeholder was too ambiguous to reverse |

`restore()` with no argument reverses `sanitized_text` itself, which round-trips
back to the original input — a cheap way to assert a policy is reversible in
tests.

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

## Handing the map to another process

`restore()` needs the `ScanResult`. When the answer comes back somewhere else —
a queue worker, another request — carry the map yourself:

```python
vault = result.token_map     # {'[EMAIL_1]': 'ali@example.com'}
```

!!! danger "The reverse map is raw PII"
    `token_map`, `restore()` and the source list all carry the original values by
    design — that is what reversibility means. Keep them on the trusted side, out
    of logs and telemetry ([`redacted()`](../reference/models.md) stays the safe
    view), and prefer a one-way action when you do not need the values back.
    Placeholder numbering is per scan, so a `ScanResult` is the only thing that
    can reverse its own text.

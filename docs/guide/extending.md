# Extending

wardcat is built around registries so you can add behaviour **without changing
the library** (Open/Closed).

## Custom actions

Actions map a detected span to its replacement (or `None` to keep the text).
Register your own — `tokenize`, `encrypt`, format-preserving masking — and use it
like any built-in:

```python
from wardcat import Wardcat, register_action

# ctx carries the salt; span has .entity_type, .text, .start, .end
register_action("tokenize", lambda span, ctx: f"<{span.entity_type}:{vault.put(span.text)}>")

guard = Wardcat(salt="s").add_entity("EMAIL", "tokenize")
```

Detection and anonymization are separate stages: `DetectionEngine` finds spans, a
standalone `Anonymizer` applies the actions — so you can reuse either independently.

## LLM backends are fixed

Unlike actions, LLM backends are **not** user-extensible. wardcat ships four —
`ollama`, `openai_compatible`, `vllm`, `transformers` — selected via the
`Backend` enum. A third-party backend would sit outside wardcat's safety checks
(the plaintext-HTTP-to-remote guard, PII handling), which is exactly where
sensitive data would leak, so point a built-in at your endpoint instead:
`openai_compatible` covers most OpenAI-style gateways (LM Studio, LocalAI,
LiteLLM, and hosted OpenAI-compatible APIs).

## Custom detectors

Every layer implements `BaseDetector` — the engine talks to detectors only
through it and never imports a concrete one. To add a whole new detector (say, a
different token-classifier model), implement `detect()` returning `DetectedSpan`s;
give model-based spans a confidence below `1.0` so a deterministic regex span
still wins overlaps.

See the [detectors & backends reference](../reference/internals.md).

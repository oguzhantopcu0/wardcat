# Extending

ai-guard is built around registries so you can add behaviour **without changing
the library** (Open/Closed).

## Custom actions

Actions map a detected span to its replacement (or `None` to keep the text).
Register your own — `tokenize`, `encrypt`, format-preserving masking — and use it
like any built-in:

```python
from ai_guard import AIGuard, register_action

# ctx carries the salt; span has .entity_type, .text, .start, .end
register_action("tokenize", lambda span, ctx: f"<{span.entity_type}:{vault.put(span.text)}>")

guard = AIGuard(salt="s").add_entity("EMAIL", "tokenize")
```

Detection and anonymization are separate stages: `DetectionEngine` finds spans, a
standalone `Anonymizer` applies the actions — so you can reuse either independently.

## Custom LLM backends

Backends are looked up in a registry, so you can add Azure OpenAI, Anthropic, or a
bespoke gateway:

```python
from ai_guard import AIGuard, BaseLLMBackend, register_backend, registered_backends

class MyBackend(BaseLLMBackend):
    def complete(self, prompt, *, timeout=60): ...
    def complete_messages(self, messages, *, timeout=60): ...
    def list_models(self): return []
    def pull_model(self, model, *, on_progress=None): ...

register_backend("my_backend", lambda cfg: MyBackend())
registered_backends()   # frozenset({"ollama", "openai_compatible", "transformers", "my_backend"})

guard = AIGuard(salt="s").with_llm(backend="my_backend", model="...")
```

## Custom detectors

Every layer implements `BaseDetector` — the engine talks to detectors only
through it and never imports a concrete one. To add a whole new detector (say, a
different token-classifier model), implement `detect()` returning `DetectedSpan`s;
give model-based spans a confidence below `1.0` so checksum-regex still wins
overlaps.

See the [detectors & backends reference](../reference/internals.md).

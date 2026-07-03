# Detection layers

An entity can be detected by one or more of **four** layers — `regex`,
`ner`, `gliner`, and `llm`. When you enable an entity it runs on every layer that
supports it; pass `layers=[...]` to target one:

```python
guard.add_entity("EMAIL", action="redact", layers=["regex"])
guard.add_entity("SPECIAL_CATEGORY", action="redact", layers=["llm"])
```

The engine merges every layer's spans and resolves overlaps **confidence-first**:
a checksum-validated regex span (`1.0`) always wins over a fuzzy NER/GLiNER/LLM
span (`≤0.99`).

Discover what each layer can detect:

```python
AIGuard.supported_entities()          # every known type
AIGuard.supported_entities("gliner")  # {"PERSON", "EMAIL", "PHONE", "IBAN", ...}
```

## Regex

Deterministic, exhaustive, and free — the backbone. 25+ patterns with
**checksum validation** (TC_ID · Nüfus İdaresi, IBAN · mod-97, CREDIT_CARD ·
Luhn) so structural PII is flagged with no false positives. Covers cards, IBAN,
SSN, NIN, TC_ID, EU national IDs, secrets/API keys, JWT, UUID, IPs, and more.
Always on for any enabled regex-supported entity; no extra dependency.

## SpaCy NER (`ner`)

Names, organisations, and locations via SpaCy. Off by default and ships no
default model — enable with a language (recommended) or an explicit model:

```python
from ai_guard import AIGuard, Language

guard = AIGuard(salt="s").with_ner(language=Language.TR).add_entity("PERSON")
guard = AIGuard(salt="s").with_ner(spacy_model=["en_core_web_sm", "de_core_news_sm"])
```

A multilingual gazetteer filters out job titles and abbreviations that NER models
commonly mislabel as names.

## GLiNER (`gliner`)

A lightweight zero-shot transformer NER (GLiNER2) — a middle ground between
SpaCy and the LLM. One multilingual model covers EN/FR/ES/DE/IT/PT/NL and a rich
PII taxonomy. Runs **as a SpaCy alternative or alongside it**. Long inputs are
chunked automatically so the model's fixed max length doesn't truncate them.

```python
guard = (
    AIGuard(salt="s")
    .with_gliner()                 # needs: pip install "ai-guard[gliner]"
    .add_entity("PERSON").add_entity("EMAIL")
)
```

!!! note "Turkish"
    The default GLiNER model is **not** trained on Turkish — keep the regex/LLM
    layers for Turkish text.

## On-prem LLM (`llm`)

The strongest context — detects semantic PII the others can't: GDPR Article 9
special categories ("HIV positive", religion…), contextual secrets
(`password=…`), unlabeled passports. Never trusted blindly: the model returns
`{"type","text"}` JSON, which is filtered by structural validators and located
back in the original text; on any failure the layer is skipped and recorded in
`ScanResult.warnings`.

```python
from ai_guard import AIGuard, Backend

# Ollama (default): needs a running Ollama service
guard = AIGuard(salt="s").with_llm(backend=Backend.OLLAMA, model="llama3.1:8b")

# In-process HuggingFace Transformers (no daemon): pip install "ai-guard[transformers]"
guard = AIGuard(salt="s").with_llm(backend=Backend.TRANSFORMERS,
                                   model="Qwen/Qwen2.5-3B-Instruct")
```

### Ensemble adjudication

With `with_llm(adjudicate=True)` the LLM verifies/relabels/drops the regex+NER
candidates **and** adds what they missed, in a single call — cleaning NER noise
(e.g. a job title mislabeled as a name). Deterministic regex spans are always
kept regardless of the LLM verdict.

See the full API on the [AIGuard reference page](../reference/aiguard.md).

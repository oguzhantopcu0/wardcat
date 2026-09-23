# Installation

The base install is deliberately tiny — `pyyaml` + `httpx`. Optional detection
layers are pulled in as extras.

=== "pip"

    ```bash
    pip install wardcat            # base: regex + Ollama/OpenAI-compatible LLM backend
    pip install "wardcat[ner]"     # + SpaCy NER (PERSON, ORG, ADDRESS)
    pip install "wardcat[phone]"   # + national phone formats (libphonenumber)
    pip install "wardcat[all]"     # everything: SpaCy + Transformers
    ```

=== "uv (from source, for development)"

    ```bash
    git clone https://github.com/oguzhantopcu0/wardcat.git
    cd wardcat
    uv sync                 # base: regex + Ollama/OpenAI-compatible LLM backend
    uv sync --extra ner     # + SpaCy NER (PERSON, ORG, ADDRESS)
    uv sync --extra all     # everything: SpaCy + Transformers
    ```

## Extras

| Extra | Adds | Layer |
|---|---|---|
| *(base)* | regex detection + Ollama / OpenAI-compatible LLM backend | regex, llm (HTTP) |
| `ner` | SpaCy | ner |
| `phone` | libphonenumber, for [`with_phone_regions()`](guide/configuration.md#phone-regions) | regex (PHONE) |
| `transformers` | HuggingFace Transformers + torch | llm (in-process) |
| `all` | everything above | all |

## SpaCy models (for the NER layer)

The NER layer needs a language model. The simplest path is to let wardcat
resolve and download it via the `language=` builder:

```python
from wardcat import Wardcat, Language

guard = Wardcat().with_ner(language=Language.EN)                   # → en_core_web_sm
guard = Wardcat().with_ner(language=Language.TR, spacy_size="md")  # → tr_core_news_md
```

Or download a model yourself:

```bash
uv run python -m spacy download en_core_web_sm
uv run python -m spacy download tr_core_news_md
```

For Turkish, prefer `tr_core_news_lg` where its size is acceptable: on the
consistency benchmark it gives one name one value across grammatical cases in
7 of 9 entities (1.2 distinct values per name), where `tr_core_news_md` runs
sentence-initial words into the name (2.9 distinct values per name).

If a requested SpaCy model is not installed, wardcat falls back to any installed
model of the same language and says so in `ScanResult.warnings`. SpaCy is not
required if you only need regex-based detection.

## Requirements

- Python **3.11+**

!!! note "Upgrading from a pre-1.0 version?"
    Each breaking change is listed with migration steps in the
    [changelog](changelog.md) — e.g. NER is configured only via `with_ner(...)`
    (0.7.0) and LLM backends are no longer user-extensible (0.9.0). As of 1.0 the
    public API is stable (semantic versioning).

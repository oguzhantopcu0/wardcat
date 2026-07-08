# Installation

> **Not yet published to PyPI.** Install from source until the first release.

The base install is deliberately tiny — `pyyaml` + `httpx`. Optional detection
layers are pulled in as extras.

=== "uv (recommended)"

    ```bash
    git clone https://github.com/oguzhantopcu0/wardcat.git
    cd wardcat
    uv sync                 # base: regex + Ollama/OpenAI-compatible LLM backend
    uv sync --extra ner     # + SpaCy NER (PERSON, ORG, ADDRESS)
    uv sync --extra all     # everything: SpaCy + Transformers
    ```

=== "pip (from Git)"

    ```bash
    pip install "git+https://github.com/oguzhantopcu0/wardcat.git"
    pip install "wardcat[ner]    @ git+https://github.com/oguzhantopcu0/wardcat.git"
    pip install "wardcat[all]    @ git+https://github.com/oguzhantopcu0/wardcat.git"
    ```

## Extras

| Extra | Adds | Layer |
|---|---|---|
| *(base)* | regex detection + Ollama / OpenAI-compatible LLM backend | regex, llm (HTTP) |
| `ner` | SpaCy | ner |
| `transformers` | HuggingFace Transformers + torch | llm (in-process) |
| `all` | everything above | all |

## SpaCy models (for the NER layer)

The NER layer needs a language model. The simplest path is to let wardcat
resolve and download it via the `language=` builder:

```python
from wardcat import Wardcat, Language

guard = Wardcat(language=Language.EN)                   # → en_core_web_sm
guard = Wardcat(language=Language.TR, spacy_size="md")  # → tr_core_news_md
```

Or download a model yourself:

```bash
uv run python -m spacy download en_core_web_sm
uv run python -m spacy download tr_core_news_md
```

## Requirements

- Python **3.11+**

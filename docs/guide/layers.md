# Detection layers

An entity can be detected by one or more of **three** layers — `regex`,
`ner`, and `llm`. When you enable an entity it runs on every layer that
supports it; pass `layers=[...]` to target one:

```python
guard.add_entity("EMAIL", action="redact", layers=["regex"])
guard.add_entity("SPECIAL_CATEGORY", action="redact", layers=["llm"])
```

The engine merges every layer's spans and resolves overlaps **confidence-first**.
Regex spans are tiered by certainty — checksum `1.0`, high-precision `0.97`,
fuzzy `0.90` — and all sit above the model layers (NER/LLM `0.85`), so a regex
match always wins an overlap and is never dropped by adjudication.

Discover what each layer can detect:

```python
Wardcat.supported_entities()          # every known type
Wardcat.supported_entities("ner")     # {"PERSON", "ORG", "ADDRESS", "LOCATION", "NRP"}
```

## Regex

Deterministic, exhaustive, and free — the backbone. 28 patterns, always on for
any enabled regex-supported entity, with no extra dependency.

**Checksum-validated, so a match is proof rather than a guess:** `TC_ID` (Nüfus
İdaresi), `IBAN` (mod-97), `CREDIT_CARD` (Luhn), `CRYPTO_WALLET` (Base58Check for
legacy Bitcoin addresses, the bech32/bech32m polymod for segwit), `NHS_NUMBER`
(mod-11), `BANK_ROUTING` (ABA mod-10 over a Federal Reserve prefix), `IMEI`
(Luhn), and every scheme inside `EU_NATIONAL_ID` — Spanish DNI/NIE check letters,
the French INSEE key, the Dutch BSN elfproef, the Polish PESEL check digit.

**Structural, matched on a distinctive shape:** email, phone, JWT, UUID, IPv4 and
IPv6, MAC address, postcodes, VAT numbers, and the provider-prefixed secrets
(`sk-`, `ghp_`, `github_pat_`, `AKIA`, `hf_`, Azure account keys, Sentry DSNs,
and more).

**Cued by the word beside them,** because they have no shape of their own: a
credential written into a sentence (`parolası ise …`, `password is …`) and
`USERNAME` (`kullanıcı adı ahmet.yilmaz`). Only the value is taken, never the
keyword, so the redacted line still reads.

Three of those checksums are weak enough that a bare digit run passes about one
time in ten — the ABA, NHS and IMEI checks. Both the cued and the bare form are
matched, and the bare one is scored at `0.70` instead, under the
[confidence floor](configuration.md#confidence-floor).

## SpaCy NER (`ner`)

Names, organisations, places and group affiliations via SpaCy — `PERSON`, `ORG`,
`ADDRESS`, `LOCATION` and `NRP`. Off by default and ships no
default model — enable with a language (recommended) or an explicit model:

```python
from wardcat import Wardcat, Language

guard = Wardcat(salt="s").with_ner(language=Language.TR).add_entity("PERSON")
guard = Wardcat(salt="s").with_ner(spacy_model=["en_core_web_sm", "de_core_news_sm"])
```

A multilingual gazetteer filters out job titles and abbreviations that NER models
commonly mislabel as names.

`LOCATION` covers countries, cities and regions, kept apart from `ADDRESS` (a
street address) because the two carry different risk. `NRP` is nationality,
religious or political affiliation — GDPR Article 9 data — and ships **off by
default**: "Turkish" and "Catholic" are ordinary vocabulary, so redacting every
occurrence would wreck the text for anyone not doing Article 9 work.

A `PERSON` normally has to contain a capitalized word. That rule is skipped where
the surrounding text carries no capitals at all, since a chat log or an ASR
transcript would otherwise lose every name to a property it never had. The trade
is that a lower-cased document can surface a common-word sequence as a `PERSON`.

### Choosing a language (and auto-detection)

One model loads per language, so NER only recognises the language(s) you select.
wardcat **does not bundle language detection** — that would add an opinion and a
dependency to a library that keeps its core to `pyyaml` + `httpx`. Instead it
exposes the selection so you can wire in *your own* detector when you need it:
detect the language with any tool you like, then pass the code. `supported_languages()`
lets you check support first:

```python
from wardcat import Wardcat, supported_languages

code = detect(text)              # your language detector of choice
if code in supported_languages():  # ('de', 'en', 'es', 'fr', 'it', 'nl', 'pt', 'tr')
    guard = Wardcat(salt="s").with_ner(language=code)
else:
    guard = Wardcat(salt="s").with_llm(...)   # LLM layer is language-agnostic
```

For genuinely mixed-language text, either pass a list (`language=["en", "de"]`,
one model each) or lean on the LLM layer, which needs no per-language model.

## On-prem LLM (`llm`)

The strongest context — detects semantic PII the others can't: GDPR Article 9
special-category data (a stated health condition, religious or political
affiliation, trade-union membership), contextual secrets (`password=…`),
unlabeled passports. It is never trusted blindly: the model returns
`{"type","text"}` JSON, which is filtered by structural validators and located
back in the original text. If the backend is unreachable the whole layer is
skipped and recorded in `ScanResult.warnings`; a transient per-chunk error
(timeout, malformed JSON) is logged and that chunk is skipped while the rest
continue.

```python
from wardcat import Wardcat, Backend

# Ollama (default): needs a running Ollama service
guard = Wardcat(salt="s").with_llm(backend=Backend.OLLAMA, model="llama3.1:8b")

# vLLM server (OpenAI-compatible API; native chat, defaults to :8000/v1)
guard = Wardcat(salt="s").with_llm(backend=Backend.VLLM,
                                   model="meta-llama/Llama-3.1-8B-Instruct",
                                   base_url="http://localhost:8000/v1")

# In-process HuggingFace Transformers (no daemon): pip install "wardcat[transformers]"
guard = Wardcat(salt="s").with_llm(backend=Backend.TRANSFORMERS,
                                   model="Qwen/Qwen2.5-3B-Instruct")
```

### Model lifecycle & choosing a backend

The **`transformers`** backend loads the model **in-process**. Weights are cached
on disk (`~/.cache/huggingface`, downloaded once), but the pipeline is loaded
into RAM/VRAM **the first time you scan** and then reused for the lifetime of
that `Wardcat` object. On a tiny 135M model the first scan pays ~3–5 s of load;
every subsequent scan in the same process is warm (~0.1 s). For an 8B model the
cold load is tens of seconds — so **where you create the `Wardcat` matters**.

#### Weight dtype

The `transformers` backend picks a dtype the device runs natively: `bfloat16` on
a CUDA card that reports support for it, `float16` on Apple Silicon, `float32` on
plain CPU. Override it when you have a reason:

```python
guard = Wardcat(salt="s").with_llm(
    backend=Backend.TRANSFORMERS, model="Qwen/Qwen2.5-3B-Instruct", dtype="float32"
)
```

The choice matters more than it looks. Apple Silicon has no bf16 arithmetic unit
— Metal emulates it, so every matmul pays a conversion the fp16 path does not —
and most CPU kernels have no half-precision path at all, falling back through
fp32 anyway. Ampere and later keep `bfloat16` by default: it runs natively there,
and its wider exponent range is the safer of the two half formats.

!!! warning "float16 on Apple Silicon"
    Some torch/transformers combinations segfault loading `float16` through
    `device_map="auto"` on MPS. If that happens, name `bfloat16` explicitly — it
    is slower there, but it loads.

The **HTTP backends** (`ollama`, `vllm`, `openai_compatible`) don't load anything
in your process — they call a daemon/server that keeps the model resident (Ollama
warms it via `keep_alive`, ~5 min; vLLM stays loaded for the server's lifetime).

!!! tip "Serving from FastAPI (or any long-lived process)"
    Create **one** `Wardcat` at startup and reuse it — the model loads once and
    stays warm for every request:

    ```python
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from wardcat import Wardcat, Backend

    guard: Wardcat | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global guard
        guard = Wardcat(salt="s").with_llm(
            backend=Backend.TRANSFORMERS, model="meta-llama/Llama-3.1-8B-Instruct"
        )
        guard.scan("warmup")   # optional: pay the cold load at startup
        yield

    app = FastAPI(lifespan=lifespan)

    @app.post("/scan")
    async def scan(text: str):
        # await the async API so concurrent requests overlap instead of blocking
        return (await guard.scan_async(text)).sanitized_text   # warm model
    ```

    **Do not** build `Wardcat(...)` inside the request handler — that reloads the
    model on every request. One shared guard is safe across concurrent scans;
    just don't *reconfigure* it while it is serving requests.

!!! warning "Multiple workers multiply VRAM"
    Each `uvicorn --workers N` / gunicorn worker is a separate process, so the
    `transformers` backend loads its **own** copy of the model — N workers ≈ N×
    VRAM. For horizontally-scaled serving use **`vllm`** (throughput) or
    **`ollama`** (simple setup) instead: every worker shares one server, so the
    weights live in VRAM **once** regardless of worker count.

**Quick guide:**

| Scenario | Backend |
| --- | --- |
| Prod serving, multiple workers, high traffic | `vllm` / `ollama` |
| Single long-lived process, batch jobs, dev, air-gapped | `transformers` |
| Repeated short-lived CLI runs | `ollama` / `vllm` (daemon stays warm between runs) |

### Async & concurrency

Every scanning call has an async twin — `scan_async`, `scan_batch_async`,
`is_sensitive_async`. CPU layers run in a thread pool; the LLM layer uses native
async I/O, so concurrent requests overlap instead of blocking:

```python
import asyncio

results = await asyncio.gather(*(guard.scan_async(t) for t in texts))
```

- A shared `Wardcat` is safe across concurrent scans (caches/detectors are
  lock-protected); don't reconfigure it while it is serving.
- `scan_async` is non-blocking, but a single Ollama on one GPU can still process
  LLM requests near-sequentially — use **vLLM** (continuous batching) or raise
  `OLLAMA_NUM_PARALLEL` for genuine parallel LLM throughput.

### Ensemble adjudication

With `with_llm(adjudicate=True)` the LLM verifies/relabels/drops the regex+NER
candidates **and** adds what they missed, in a single call — cleaning NER noise
(e.g. a job title mislabeled as a name). Deterministic regex spans are always
kept regardless of the LLM verdict.

### Semantic sensitivity gate — `is_sensitive()`

For a yes/no guardrail rather than per-entity extraction, `Wardcat.is_sensitive(text)`
returns a single boolean: does the text contain sensitive information (PII,
credentials, financial, special-category, or confidential business data)? It is
a holistic LLM judgement, so it also flags things the typed detectors don't —
unreleased financials, deal terms, a confidential project.

```python
guard = Wardcat().with_llm(model="gemma3:12b")
if guard.is_sensitive(user_text):   # or: await guard.is_sensitive_async(...)
    raise ValueError("won't forward sensitive text")
```

LLM-only (no entities to enable); requires `with_llm(...)`; empty text is `False`.
Fail-closed — a backend error propagates rather than returning a misleading `False`.

See the full API on the [Wardcat reference page](../reference/wardcat.md).

# Changelog

All notable changes to `wardcat` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.5.0] — 2026-07-04

> **⚠ BREAKING — the project was renamed `ai-guard` → `wardcat`.** The import
> package is now `wardcat` (`from wardcat import Wardcat`), the main class
> `AIGuard` is now **`Wardcat`**, and the distribution and its extras are
> `wardcat` / `wardcat[ner]` / `wardcat[gliner]` / `wardcat[transformers]` /
> `wardcat[all]`. Example env-var names in the docs are now `WARDCAT_*`. Update
> your imports and install commands — no runtime behaviour changed.

### Added

- **GLiNER zero-shot NER layer.** A new detector layer wraps the PII-tuned **GLiNER2** model (`fastino/gliner2-privacy-filter-PII-multi`, Apache-2.0) — a lightweight bidirectional-encoder NER that sits between SpaCy NER and the LLM. Enable it with the chainable `with_gliner()` builder (mirrors `with_ner()`/`with_llm()`); entity types are opt-in via `add_entity(...)`. It runs as a **SpaCy alternative or alongside SpaCy** — the engine merges both layers' spans and a checksum-validated regex span always wins an overlap (GLiNER spans are capped at 0.99 confidence). The new `"gliner"` layer is selectable via `layers=["gliner"]` and `supported_entities("gliner")`. Ships as an optional extra — `pip install "wardcat[gliner]"` (pulls in torch via `gliner2[local]`); the base install stays `pyyaml` + `httpx`. The default model covers EN/FR/ES/DE/IT/PT/NL (not Turkish — keep the regex/LLM layers for Turkish text). **Long inputs are automatically chunked** (`chunk_size`, default 1500 chars) so the model's fixed maximum length does not silently truncate long documents. Configurable via YAML under `gliner_detector:` (`enabled`, `model`, `threshold`, `quantize`, `chunk_size`).
- **Degraded-scan visibility — `ScanResult.warnings`.** When a detector layer cannot run (most commonly the LLM backend being unreachable), the scan now records the failure on `result.warnings` (and in `redacted()`) instead of silently swallowing it. The other layers still run and return results, but a non-empty `warnings` list tells the caller detection was degraded — no more thinking the LLM ran when it never connected. A backend `ConnectionError` now propagates from the LLM detector so the engine can surface it uniformly for any layer.
- **Value propagation (`with_propagation()`).** Once *any* layer detects a value, every other whole-token occurrence of it in the text is anonymized too — closing the gap where a model-based layer (GLiNER/NER/LLM) reports a repeated value only once (verified: a name GLiNER caught 2 of 3 times leaked one occurrence; with propagation, 0 leaked). Off by default (it can over-redact); only exact, token-bounded matches ≥ `min_length` chars (default 3) propagate, and deterministic regex spans still win overlaps. Config keys: `propagate_matches`, `propagate_min_length`.

### Fixed

- **Turkish `ADDRESS` regex over-capture.** The pattern grabbed 1–5 words of *any* case before the street-type keyword, so it swallowed lowercase filler and crossed sentence boundaries (e.g. `"iletişime geçilebilir. İkamet adresi Bağdat Caddesi"`) while missing the `No:`/`Daire:` tail. It now takes only 1–3 **capitalized** preceding words and optionally captures a `No:`/`Daire:`/`Kat:` suffix — `"… Bağdat Caddesi No:127 Daire:8"` is captured cleanly.
- **`DATE_OF_BIRTH` is now an LLM entity.** It was only a `regex`/`gliner` entity, so the LLM layer was never asked for birth dates — a date the regex missed (e.g. `"14.03.1985 doğumlu"`, keyword *after* the date) fell through both active layers. Added a `DATE_OF_BIRTH` description to the LLM prompt so the LLM catches contextual birth dates. (The prompt already had few-shot examples for it.)
- **Transformers backend real inference (`dtype` → `torch_dtype`).** The HuggingFace pipeline was built with a `dtype=` kwarg, which only exists in transformers ≥ 4.56; on the pinned `>=4.40,<5` range it was silently forwarded to `generate()` and rejected — so real inference failed with *"model_kwargs are not used: ['dtype']"* on every call. The backend now uses `torch_dtype` (valid across the whole range). This was never caught because the backend's tests are mock-only and it had not been run against a real model; a regression test now asserts the pipeline is built with `torch_dtype`.

### Removed

- **The command-line interface is gone.** The `wardcat` console script and the `python -m wardcat` entry point (`scan`, `batch`, `spacy`, `models` sub-commands) were removed, along with the `[project.scripts]` entry point and the `WARDCAT_*` environment-variable handling that only the CLI used. `wardcat` is a **library**; drive it from Python (`from wardcat import Wardcat`). Manage models with their native tools instead: SpaCy models via `python -m spacy download <model>` (or the `language=` builder, which auto-downloads), and on-prem LLMs via `ollama pull <model>` (or `with_llm(model=..., auto_pull=True)`).

### Changed

- **Overlap resolution is confidence-first and robust to chained overlaps.** When detected spans overlap, the engine now keeps the strongest span — highest confidence first (a checksum/regex `1.0` span beats a longer fuzzy NER/LLM `0.85` span), then longest, then earliest — instead of blindly keeping the longest. Every candidate is checked against **all** already-kept spans, closing a gap where a chained/nested overlap could let a span slip through. This prevents a Luhn-validated card from being lost to an overlapping `ADDRESS` guess.

---

## [0.4.0] — 2026-06-21

> Includes **breaking changes** (class and method renames). Deprecated aliases are kept for one release cycle.

### Breaking

- **LLM is configured via `with_llm()` (or YAML), not constructor arguments.** The `use_llm`, `llm_backend`, `llm_model`, `llm_base_url`, `llm_api_key`, `llm_timeout`, `llm_allow_http`, `llm_adjudicate`, `auto_pull`, `llm_device_map`, `llm_load_in_8bit`, `llm_load_in_4bit` constructor parameters were removed — use `Wardcat(...).with_llm(backend=..., model=..., ...)`. This slims the constructor (19 → 7 params) and removes the duplicate path. NER constructor args (`use_ner`, `language`, `spacy_model`, …) are unchanged.
- **Detection is opt-in: a bare `Wardcat()` starts empty.** Previously every regex/NER entity was on by default; now nothing is enabled until you `add_entity()` / `add_entities()`. Enable everything with `add_entity(Entity.ALL, action=...)`. (The `wardcat` CLI keeps a sensible "detect common PII" default policy, since it is an application.) The unsalted-hash warning now fires from the first rebuild that activates a `hash` action, not only at construction.
- **`LLMGuard` → `Wardcat`:** the main class is renamed and the `LLMGuard` name is **removed** (the guard is not LLM-specific — it is regex/NER/LLM hybrid). Update imports to `from wardcat import Wardcat`.
- **`configure_entity()` → `add_entity()`** and **`configure_entities()` → `add_entities()`.** The old method names were **removed** (no aliases).
- **`add_entity()` / `add_entities()` no longer take an `enabled` argument.** Adding an entity always enables it; use `remove_entity()` / `remove_entities()` to turn entities off. This removes the contradictory `add_entity(..., enabled=False)` form.
- **Default action is now `hash` (was `warn`).** Calling `add_entity()` / `add_entities()` without an `action` enables the entity with `action="hash"` (the safest default) and logs a warning; pass `action=...` explicitly to silence it.
- **NER is off by default and ships no default model.** `use_ner` now defaults to off, and the old `spacy_model="en_core_web_sm"` default is gone (constructor and `default.yaml`). Enable NER explicitly with `language=...` (recommended) or `spacy_model=...`; `use_ner=True` without a model raises `ConfigError`. A named-but-missing model is auto-downloaded.
- **The library no longer reads environment variables.** `load_config()` / `Wardcat()` ignore the environment entirely — pass configuration explicitly via constructor arguments or a YAML `config_path`. Reading env vars is now confined to the **`wardcat` CLI** (an application), where `WARDCAT_SALT`, `WARDCAT_LLM_URL`, `WARDCAT_LLM_MODEL`, `WARDCAT_LLM_API_KEY` act as defaults for the matching flags. (The old `LLMGUARD_*` names are gone.)
- **HTTP-to-remote-LLM override is a parameter, not an env var:** the `LLMGUARD_ALLOW_HTTP` env var was removed; pass `Wardcat(llm_allow_http=True)` (or `allow_http=` on a backend) to permit plaintext HTTP to a remote host (still blocked by default).

### Added

- **`Entity` constants:** a new `Entity` enum exposes every known entity type as a constant (`Entity.CREDIT_CARD`, `Entity.EMAIL`, …) for IDE autocomplete and typo-proof configuration. Use it anywhere a string entity type was accepted — `guard.add_entity(Entity.CREDIT_CARD, action=Action.HASH)`. Bare strings still work; `Entity` *is* its string value (`Entity.EMAIL == "EMAIL"`).
- **`Entity.ALL` sentinel:** `add_entity(Entity.ALL)` enables **every** known entity type in one call (and `add_entities([Entity.ALL, ...])` expands it inline). It is excluded from `KNOWN_ENTITY_TYPES`. (`Entity.All` is kept as a deprecated PascalCase alias.)
- **`remove_entity()` / `remove_entities()`:** disable one or many entity types (across all detector layers). `remove_entity(Entity.ALL)` disables everything. The natural pattern is "enable all, then prune": `guard.add_entity(Entity.ALL, action="hash").remove_entity(Entity.ORG)`. Removing an entity that was never enabled is a no-op; an unknown *name* logs a warning (like `add_entity`) to catch typos.
- **`change_entity_action()`:** retarget the action of an entity that is **currently enabled** without changing its layers — `guard.change_entity_action(Entity.EMAIL, Action.HASH)`. It refuses to silently re-enable: changing the action of a removed or never-added entity raises `ConfigError` (enable it first with `add_entity()`). `change_entity_action(Entity.ALL, ...)` changes the action of every currently-enabled entity.
- **Introspection:** `enabled_entities()` returns the set of currently-enabled entity types; `get_entity_action(entity)` returns an entity's action (or `None` if it is not enabled); `entity_policy()` returns the full `{entity: action}` mapping. This rounds out the write API (add/remove/change) with a read API.
- **Pluggable LLM backends (Open/Closed):** a backend registry replaces the hard-coded `if/elif` backend selection. Register a custom backend without touching the core — `register_backend("name", factory)` — then use it via `with_llm(backend="name")`. `BaseLLMBackend`, `register_backend`, and `registered_backends` are exported from `wardcat`; backend validation now reflects the live registry.
- **Pluggable actions (Open/Closed):** anonymization actions live in a registry instead of a hard-coded `if/elif`. Register a custom action — `register_action("tokenize", lambda span, ctx: ...)` — and use it like any built-in (`add_entity("EMAIL", "tokenize")`). Built-ins (`warn`/`hash`/`redact`/`mask`) are registered the same way; `register_action`, `registered_actions`, and `ActionContext` are exported. Action validation reflects the live registry.
- **Detection ⊥ anonymization split:** action application moved out of `DetectionEngine` into a separate `Anonymizer` (analysis finds spans → anonymization transforms them), mirroring the analyze/anonymize split of mature PII pipelines. `Violation.action` is now the action **name** (`str`); it still compares equal to the matching `Action` constant (`v.action == Action.HASH`).
- **Discoverability:** `Wardcat.supported_entities(layer=None)` returns the entity types wardcat can detect — all of them, or just one layer's set (`"regex"` / `"ner"` / `"llm"`).
- **Typed `redacted()`:** `ScanResult.redacted()` now returns a `RedactedResult` `TypedDict` (with `RedactedViolation` items), both exported from `wardcat`, so the safe-logging payload has a precise, IDE-visible shape.
- **`Language` constants:** a new `Language` enum (`Language.EN`, `DE`, `FR`, `ES`, `IT`, `NL`, `PT`, `TR`) for documented, typo-proof NER language selection — `Wardcat(language=Language.DE)` or a list for multilingual NER. Plain ISO codes are still accepted.
- **`Backend` constants:** a new `Backend` enum (`Backend.OLLAMA`, `Backend.OPENAI_COMPATIBLE`, `Backend.TRANSFORMERS`) for typo-proof LLM backend selection — `Wardcat(llm_backend=Backend.OPENAI_COMPATIBLE)`. Plain strings still work; `_VALID_BACKENDS` is now derived from the enum.
- **Fluent layer builders `with_ner()` / `with_llm()`:** a chainable alternative to the wide constructor that keeps each layer's settings in one place and makes NER/LLM symmetric — `Wardcat(salt="s").with_ner(language=Language.TR).with_llm(backend=Backend.OLLAMA, model="llama3.2")`. Both return `self` and can be chained back-to-back. The constructor `llm_*` / `spacy_*` arguments still work.

### Fixed

- **Static-analysis pass (bandit / radon / pip-audit):** flag the LLM cache-key MD5 as `usedforsecurity=False` (non-cryptographic, was bandit's only High); split the high-complexity `validate_config` (cyclomatic 32 → 1, via per-section helpers); bump vulnerable dev/test dependencies (idna, urllib3, requests, pytest, pygments) — `pip-audit` now reports no known vulnerabilities. Coverage 93%, no dead code.
- **Broken `examples/batch_and_async.py`:** it scanned without enabling any entity, so after the opt-in change it silently reported everything as clean. Examples now enable entities, and a new smoke test (`tests/test_examples.py`) runs the offline examples in CI and asserts they actually detect PII — so examples can't rot silently again.
- **Default-action warning noise:** when `add_entity()` / `add_entities()` default a missing action to `hash`, the warning is now logged **once per guard** instead of on every call — it stays visible without spamming logs when many entities are added.
- **Misleading install hint:** the LLM backends' missing-`httpx` error pointed at a non-existent `wardcat[llm]` extra; `httpx` is a core dependency, so the message now says to reinstall wardcat.
- **Multiple explicit models:** `spacy_model=` now accepts a list, e.g. `spacy_model=["en_core_web_sm", "de_core_news_sm"]`.
- **Salt:** when no salt is set and a `hash` action is in play, wardcat logs a clear warning (rainbow-table risk) and proceeds with unsalted hashes; set `salt=...` or `WARDCAT_SALT`.
- `Entity` and `Action` are first-class, type-hinted arguments (`entity_type: str | Entity`, `action: str | Action`). Static type checkers flag an invalid action or entity at edit time instead of at runtime.

### Changed

- `KNOWN_ENTITY_TYPES` is now derived from the `Entity` enum (excluding the `Entity.ALL` sentinel) — the enum is the single source of truth, so the two can no longer drift apart.
- `add_entity()` / `add_entities()` normalize `Entity`/`Action` enum arguments to their canonical string form before storing them in the config.
- **Error handling:** the entity-management API (`add`/`remove`/`change`/`get`) now raises `ConfigError` for a non-`str`/`Entity` entity argument, an invalid/wrong-typed action, an unknown layer, a malformed `add_entities()` argument (bare string or non-iterable), or `get_entity_action(Entity.ALL)`. Unknown *entity names* still warn (not raise) so custom entity types keep working.

---

## [0.3.0] — 2026-06-20

> Includes a **breaking change** (removal of the shipped ASGI/FastAPI middleware).

### Added

- **Layer-aware filter selection:** `configure_entity(entity, layers=[...])` targets a specific detector layer (`"regex"`, `"ner"`, `"llm"`); when omitted, every layer that supports the entity is used. Lets you keep semantic-only entities (e.g. `SPECIAL_CATEGORY`) off the regex/NER path.
- **Batch filter configuration:** `configure_entities()` enables many entity types in one call (single rebuild). Accepts a list, a `{name: action}` mapping, or a `{name: {action, layers, enabled}}` mapping, and pairs with the predefined entity groups (`turkish_entities()`, `european_entities()`, …).
- **NER model selection by language:** `LLMGuard(language="de", spacy_size="md")` resolves the SpaCy model from the catalog by language code and size tier (`sm`/`md`/`lg`/`trf`); supported languages are `en`, `de`, `fr`, `es`, `it`, `nl`, `pt`, `tr`. Selecting a language implies auto-download of the model if it is missing (disable with `spacy_auto_download=False`).
- **Multilingual NER:** `LLMGuard(language=["en", "de", "fr"])` loads one NER detector per language; the engine merges their spans. Each model loads independently (one failure skips only that model). Explicit by design — wardcat does not auto-detect the input language.
- **Reusable SpaCy installer:** new `wardcat.ner.downloader` module (`ensure_model`, `download_model`) shared by the CLI and the auto-download path; `wardcat.ner.spacy_catalog.resolve_model()` resolves a language + size to a catalog model.
- **CLI language flags:** `--lang`, `--spacy-size`, and `--spacy-auto-download` on `wardcat scan` / `wardcat batch`.
- **Ensemble adjudication:** opt-in `LLMGuard(llm_adjudicate=True)` / `llm_detector.adjudicate` — the LLM verifies, relabels, drops, and supplements regex/NER candidates in a single call. Deterministic regex matches are always kept; LLM-only mode is unaffected.
- **GDPR Article 9 detection:** new `SPECIAL_CATEGORY` entity (health, religion, ethnicity, political opinion, sexual orientation, trade-union, genetic/biometric). LLM-only and **off by default** (semantic, subjective); enable under `llm_detector.entities`.
- **`VAT_NUMBER` entity:** EU-prefixed VAT IDs (DE/FR/GB/IT/ES/AT/NL) and the Turkish *Vergi No* keyword form.
- **Multilingual filters:** `DATE_OF_BIRTH` (DE/FR month names and keywords), `PHONE` (French national, German mobile), and LLM prompt + few-shot examples extended to EN/DE/FR/TR.
- **Expanded `CUSTOM_SECRET`:** Stripe (`sk_live_`, `rk_live_`), Anthropic (`sk-ant-`), Google (`AIza`), GitLab (`glpat-`), SendGrid (`SG.`), Twilio (`SK`/`AC`), npm, Slack webhook URLs, and PEM private-key blocks.
- **Connection-string credential detection:** passwords in `scheme://user:pass@host` URIs are flagged as `CUSTOM_SECRET` (and the spurious EMAIL match they used to trigger is suppressed).
- **Tooling:** `ruff` (lint + format) and `mypy` configured in `pyproject.toml`, with a "Lint & type-check" CI job that the test job depends on.
- **Live LLM integration tests:** `tests/integration/test_llm_live.py` runs against a real Ollama model (`@pytest.mark.slow`, auto-skips when Ollama is unavailable).
- **Docs & examples:** `CONTRIBUTING.md`; `examples/` (`asgi_middleware.py`, `batch_and_async.py`, `llm_hybrid.py`, `demo.py`).

### Changed

- **`CREDIT_CARD` validation:** now Luhn-checked via a table-driven `_VALIDATORS` registry (alongside TC_ID and IBAN checksums).
- **`requires-python`:** aligned to `>=3.11`.

### Removed

- **BREAKING — shipped ASGI/FastAPI middleware:** the `wardcat.integrations` package was removed; the core is now a pure detection library (deps stay `pyyaml` + `httpx` only). A self-contained, copy-paste ASGI middleware now lives in `examples/asgi_middleware.py`.

### Fixed

- **`US_ZIP_CODE` ZIP+4 leak:** a labeled branch grabbed only the first 5 digits of a ZIP+4; fixed with a negative lookahead.
- **`FINANCIAL_AMOUNT`:** wired into the guard's regex entity set (previously dead code); remains opt-in/off by default.

---

## [0.2.0b1] - 2026-03-20

### Added

- **New entity types:** `UK_POSTAL_CODE` (British postcodes), `US_ZIP_CODE` (ZIP+4 format), and `EU_NATIONAL_ID` (Spanish DNI and NIE) via regex detection.
- **Multilingual ADDRESS patterns:** French (Rue, Allée, Boulevard…), Spanish (Calle, Avenida, Plaza…), Italian (Piazza, Corso, Via…), Dutch (straat, gracht, laan…), and German (Straße, Weg, Platz…) street-type keywords added to the address regex.
- **`PASSPORT` entity:** LLM-based contextual detection for passport numbers of any country. Requires `use_llm=True`.
- **`CUSTOM_SECRET` entity:** Regex detection for known API token prefixes — `sk-`, `ghp_`, `AKIA`, `ya29.`, `xoxb-`, `xoxp-`. LLM layer extends this to contextual secrets (`password=VALUE`, `api_key=VALUE`).
- **`ScanResult.redacted()`:** Returns a PII-free dict suitable for safe logging and API responses without exposing raw PII from `original_text` or `violations[].original`.
- **SpaCy model auto-fallback:** When the requested SpaCy model is not installed, wardcat automatically falls back to any installed model of the same language and emits a warning.
- **DoS protection:** Inputs exceeding 500 KB now raise a `ValueError`. Previously this was a warning only.
- **HTTP warning on all LLM backends:** A security warning is logged for HTTP connections to any LLM backend, including localhost. Use HTTPS via a reverse proxy in production.
- **CLI salt warning:** The `wardcat scan` and `wardcat batch` commands warn when the `hash` action is used without a salt, indicating rainbow table vulnerability.
- **GitHub Actions CI:** Matrix testing across Python 3.11, 3.12, and 3.13; enforces ≥80% test coverage; includes wheel build verification.

### Changed

- **Hash digest length:** Upgraded from 8 hex characters (32-bit entropy) to 16 hex characters (64-bit entropy). Replacement tokens now appear as `[TYPE:ea782818c5a992a8]`.
- **TC_ID validation:** Now validated with the official Nüfus İdaresi checksum algorithm, eliminating false positives from random 11-digit sequences.
- **IBAN validation:** Now validated with the ISO 13616 mod-97 checksum algorithm before flagging.
- **ADDRESS regex:** Tightened pattern to reduce false positives on common non-address text.
- **IPv6 validator:** Replaced previous implementation with a proper RFC 5952 alternation regex covering full and compressed forms.
- **Version sourcing:** Package version is now read from `importlib.metadata` at runtime (single source of truth from `pyproject.toml`).

### Fixed

- **Transformers backend:** Chat template availability check moved to the correct location in the inference pipeline.
- **SpaCy NER fallback:** Warning message wording made consistent across all fallback code paths.

[Unreleased]: https://github.com/oguzhantopcu0/wardcat/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/oguzhantopcu0/wardcat/compare/v0.2.0b1...v0.3.0
[0.2.0b1]: https://github.com/oguzhantopcu0/wardcat/releases/tag/v0.2.0b1

---

## [0.2.0] — 2026-03-19

### Added
- **6 new entity types** — `UUID`, `SSN` (US), `MAC_ADDRESS`, `JWT`, `IPv6`, `NIN` (UK) with regex detection
- **International phone support** — E.164 format (`+1`, `+44`, etc.) added alongside Turkish phone patterns
- **International address support** — English street address patterns (Street, Avenue, Road, etc.) alongside Turkish patterns
- **HuggingFace Transformers backend** — on-prem GPU/CPU inference via `transformers` pipeline; supports Llama 3.1/3.2 (1B, 3B, 8B, 70B), 8-bit/4-bit quantization, `device_map="auto"` for multi-GPU
- **`build_messages()`** — chat-format message builder for backends with native chat support
- **`BaseLLMBackend.complete_messages()`** — all backends now support chat message format with default fallback
- **Structural validators for new entity types** — hallucination filtering for UUID, SSN, MAC_ADDRESS, JWT, IPv6, NIN
- **LLM detector covers all entity types** — `llm_detector.entities` config now includes all 16 entity types including POSTAL_CODE

### Changed
- `LLMDetector` now uses `complete_messages()` instead of `complete()` — better prompt formatting for chat-capable models
- `ModelInfo` now has a `backend` field (`"ollama"` or `"transformers"`)
- Model catalog expanded with HuggingFace Llama model IDs

[0.2.0]: https://github.com/oguzhantopcu0/wardcat/compare/v0.1.0...v0.2.0

---

## [0.1.0] — 2026-03-19

Initial release.

### Added

- **Hybrid detection engine** — regex + SpaCy NER + on-prem LLM (Ollama / OpenAI-compatible)
- **Regex detectors** — `CREDIT_CARD`, `EMAIL`, `PHONE`, `IBAN`, `TC_ID`, `IP_ADDRESS`, `ADDRESS`, `POSTAL_CODE`
- **NER detector** — `PERSON`, `ORG`, `ADDRESS` via SpaCy (English `en_core_web_sm` and Turkish `tr_core_news_*` models)
- **LLM detector** — `CUSTOM_SECRET` and any entity type via Ollama or OpenAI-compatible backends
- **Two actions** — `warn` (report, keep text) and `hash` (replace with `[TYPE:8hex]` using salted SHA-256)
- **Python API** — `LLMGuard`, `ScanResult`, `Violation`, `Action`; method-chaining configuration
- **YAML API** — declarative policy files with per-entity enable/action settings
- **`scan_batch()`** — fault-isolated batch scanning (one failure doesn't abort the batch)
- **CLI** — `wardcat scan`, `wardcat batch`, `wardcat models list/setup/pull`
- **Environment variable overrides** — `LLMGUARD_SALT`, `LLMGUARD_LLM_URL`, `LLMGUARD_LLM_MODEL`, `LLMGUARD_LLM_API_KEY`, `LLMGUARD_LLM_TIMEOUT`, `LLMGUARD_SPACY_MODEL`
- **Config validation** — `validate_config()` checks actions, backend names, timeout values
- **SpaCy singleton cache** — thread-safe model cache prevents reloading 300-500 MB models per instance
- **Hallucination filtering** — structural validators per entity type (e.g. PERSON requires ≥2 words, TC_ID must be exactly 11 digits)
- **Overlap resolution** — longer span wins when two detectors produce overlapping matches
- **PEP 561 compliance** — `py.typed` marker for typed library consumers
- **Optional extras** — `[ner]` for SpaCy, `[transformers]` for HuggingFace, `[all]` for everything
- **ReDoS protection** — adversarial input tests with 30-second timeout guardrails
- **Thread-safety** — concurrent scan tests validating shared-state safety

[0.1.0]: https://github.com/oguzhantopcu0/wardcat/releases/tag/v0.1.0

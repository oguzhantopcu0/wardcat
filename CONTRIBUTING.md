# Contributing to wardcat

Thanks for your interest in improving wardcat! This guide covers local setup,
the quality gates, and conventions.

## Development setup

wardcat uses [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/oguzhantopcu0/wardcat.git
cd wardcat
uv sync --dev
```

This installs the library, the dev tools (pytest, ruff, mypy), and the English
SpaCy model (`en_core_web_sm`).

### Optional models

```bash
# Turkish NER (used by some tests)
uv run python -m spacy download tr_core_news_md

# Live LLM tests (optional) — install Ollama, then pull a small model
ollama pull llama3.2:1b
```

> **Note:** `uv sync` removes manually-installed SpaCy models that aren't in the
> lockfile (e.g. the Turkish HuggingFace wheels). Reinstall them after a sync with
> `UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install --no-deps <wheel-url>`.

## Quality gates

All four must pass before a PR is merged — CI enforces them.

```bash
uv run ruff check .            # lint
uv run ruff format --check .   # formatting
uv run mypy                    # type check (src/wardcat)
uv run pytest -m "not slow"    # tests (fast)
```

Auto-fix lint and apply formatting:

```bash
uv run ruff check . --fix
uv run ruff format .
```

## Running tests

```bash
uv run pytest                       # everything (live LLM tests skip if no Ollama)
uv run pytest -m "not slow"         # fast — skip live LLM tests
uv run pytest -m ner                # SpaCy NER tests only
uv run pytest -m slow tests/integration/test_llm_live.py   # live LLM, real model
uv run pytest --cov=src/wardcat --cov-report=term-missing # coverage
```

- **Mocked LLM tests** (default) verify plumbing — no model required.
- **Live LLM tests** (`slow`) call a real Ollama model and auto-skip when it is
  unavailable. Choose the model with `WARDCAT_TEST_LLM_MODEL=<name>`.
- **README examples** (`tests/unit/test_readme_examples.py`) run every code
  block in the README, compare the printed output with the `# …` comments, and
  resolve every link against `docs/` and the repository. Change the README and
  the docs together.

### Detection quality

`tests/benchmark/` scores the detectors two ways: a false-positive suite
(detectors must stay quiet on clean text) and a precision/recall harness over a
labelled, checksum-valid corpus. Both run in CI. Print the P/R report:

```bash
uv run python -m tests.benchmark.eval_harness
```

Widen coverage by adding rows to `CORPUS` in `tests/benchmark/eval_harness.py`.
`benchmarks/` holds the reproducible comparison against Microsoft Presidio on
public corpora; its README says how to run it and where the data comes from.

### Docs

The site at <https://docs.wardcat.com> is built from `docs/` with MkDocs +
Material and an auto-generated API reference. Preview it locally:

```bash
uv run --group docs mkdocs serve
```

CI builds it with `--strict`, so a broken link or a page missing from the
`mkdocs.yml` nav fails the build.

## Repository layout

```
wardcat/
├── src/wardcat/
│   ├── guard.py              # Wardcat — main interface, layer builders, classify()
│   ├── _entity_policy.py     # add/remove/change entity + introspection (mixin)
│   ├── entity_groups.py      # core_entities(), turkish_entities(), … helpers
│   ├── presets.py            # kvkk / gdpr / pci_dss / hipaa_lite / secrets_only
│   ├── cli.py                # the `wardcat` command
│   ├── exceptions.py         # WardcatError and friends (ConfigError, DegradedScanError…)
│   ├── core/
│   │   ├── engine.py         # DetectionEngine — overlap resolution, layer merge, denylist
│   │   ├── anonymizer.py     # applies the action to each resolved span
│   │   ├── actions.py        # action registry — hash/redact/mask/warn/tokenize/surrogate
│   │   ├── restore.py        # TokenAllocator + restore() — the reversible path
│   │   ├── registry.py       # which entity types each layer can produce
│   │   └── models.py         # Entity, Action, Layer, Violation, ScanResult, SensitivityVerdict
│   ├── detectors/
│   │   ├── base.py           # BaseDetector ABC
│   │   ├── regex_detector.py # patterns, checksum validators, keyword-cued secrets/usernames
│   │   ├── ner_detector.py   # SpaCy NER (multilingual) + gazetteer FP filter + span cleanup
│   │   └── llm_detector.py   # LLM-based detection with hallucination filter, circuit breaker
│   ├── llm/
│   │   ├── backends/         # ollama, openai_compat, vllm, transformers + registry
│   │   ├── circuit.py        # CircuitBreaker
│   │   ├── model_catalog.py  # supported model list
│   │   ├── model_manager.py  # download / cache lifecycle
│   │   └── prompt.py         # detection, sensitivity and classification prompts
│   ├── ner/
│   │   ├── spacy_catalog.py  # language + size tier → SpaCy package name
│   │   └── downloader.py     # auto-download of missing models
│   ├── surrogates/           # SurrogateAllocator + per-locale value pools
│   ├── config/
│   │   └── loader.py         # YAML loader and validation (no env lookup)
│   └── utils/
│       ├── hashing.py        # SHA-256 + salt
│       ├── logsafe.py        # describe() — log a value's length, never its text
│       ├── normalize.py      # confusable / homoglyph folding
│       ├── regex_safety.py   # ReDoS screen for custom and denylist patterns
│       └── text.py           # chunking and offset helpers
├── tests/
│   ├── unit/                 # component-level tests
│   ├── integration/          # scenario and adversarial tests
│   └── benchmark/            # eval harness and the false-positive suite
├── benchmarks/               # Presidio comparison (data and results are not committed)
├── examples/                 # runnable scripts
├── docs/                     # the documentation site
├── config/
│   └── default.yaml          # example policy file
└── pyproject.toml
```

## Architecture in one minute

Three detector layers feed a single engine:

- **Regex** (`detectors/regex_detector.py`) — structural PII, with checksum
  validators in the `_VALIDATORS` registry. Confidence is tiered by the evidence:
  `1.00` checksum-verified, `0.97` distinctive structure, `0.90` keyword-cued
  heuristic, `0.70` a checksum match with no supporting keyword.
- **NER** (`detectors/ner_detector.py`) — SpaCy names, organisations, places and
  group affiliations (0.85).
- **LLM** (`detectors/llm_detector.py`) — contextual/semantic PII (0.85); can
  also adjudicate the other layers' candidates in one call.

The `DetectionEngine` (`core/engine.py`) merges spans, resolves overlaps (higher
confidence wins, then the longer span), drops spans below `min_confidence`,
applies allow/deny lists, and hands the result to the `Anonymizer`, which runs
the configured action (`warn` / `hash` / `redact` / `mask` / `tokenize` /
`surrogate`, or one added with `register_action`).

A layer that cannot run must say so in `ScanResult.warnings` — failing mid-scan
through `_safe_detect`, failing while the guard is built through
`BaseDetector.build_warnings` or the `build_warnings` the guard hands the engine.
Logging alone is not enough: nobody reads a log line before trusting a clean
result.

## Adding a new entity type

1. **Regex entity:** add a pattern to `_PATTERNS` in `regex_detector.py`
   (+ a validator in `_VALIDATORS` if it has a checksum), add a member to
   `Entity` in `core/models.py` (which feeds `KNOWN_ENTITY_TYPES`), list it in
   `REGEX_ENTITIES` in `core/registry.py`, and document it in `default.yaml`.
   Entities are opt-in, so there is no default-on entry to add.
2. **LLM-only entity:** add a description (+ example) to `llm/prompt.py`, an
   entry under `llm_detector.entities` in `config/loader.py`, and an `Entity`
   member.
3. Add it to the matching groups in `entity_groups.py`, then add tests and a row
   to the entity table in `docs/reference/entities.md`.

## Conventions

- Match the surrounding style; `ruff format` is the source of truth.
- Add type hints to new public functions (the package ships `py.typed`).
- Prefer small, deterministic regex with validators over broad patterns.
- Keep hard dependencies minimal — heavy backends (SpaCy, Transformers) stay
  behind optional extras.
- Update `CHANGELOG.md` (`[Unreleased]`) for user-facing changes.

## Pull requests

- One logical change per PR; keep formatting-only churn in its own commit.
- Ensure all four quality gates pass locally.
- Describe the change and how you verified it.

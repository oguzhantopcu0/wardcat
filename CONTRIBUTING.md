# Contributing to ai-guard

Thanks for your interest in improving ai-guard! This guide covers local setup,
the quality gates, and conventions.

## Development setup

ai-guard uses [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/oguzhantopcu0/ai-guard.git
cd ai-guard
uv sync --dev
```

This installs the library, the dev tools (pytest, ruff, mypy), and the English
SpaCy model (`en_core_web_sm`).

### Optional models

```bash
# Turkish NER (used by some tests)
uv run python -m ai_guard spacy download tr_core_news_md

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
uv run mypy                    # type check (src/ai_guard)
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
uv run pytest --cov=src/ai_guard --cov-report=term-missing # coverage
```

- **Mocked LLM tests** (default) verify plumbing — no model required.
- **Live LLM tests** (`slow`) call a real Ollama model and auto-skip when it is
  unavailable. Choose the model with `AIGUARD_TEST_LLM_MODEL=<name>`.

## Architecture in one minute

Three detector layers feed a single engine:

- **Regex** (`detectors/regex_detector.py`) — structural PII, deterministic
  (confidence 1.0), with checksum/Luhn validators in the `_VALIDATORS` registry.
- **NER** (`detectors/ner_detector.py`) — SpaCy names/orgs/locations (0.85).
- **LLM** (`detectors/llm_detector.py`) — contextual/semantic PII (0.85); can
  also adjudicate the other layers' candidates in one call.

The `DetectionEngine` (`core/engine.py`) merges spans, resolves overlaps
(longest wins), applies allow/deny lists, and runs the configured action
(`warn` / `hash` / `redact` / `mask`).

## Adding a new entity type

1. **Regex entity:** add a pattern to `_PATTERNS` in `regex_detector.py`
   (+ a validator in `_VALIDATORS` if it has a checksum), then register it in
   `guard._REGEX_ENTITIES`, `loader.DEFAULT_CONFIG["entities"]`, `default.yaml`,
   and `models.KNOWN_ENTITY_TYPES`.
2. **LLM-only entity:** add a description (+ example) to `llm/prompt.py` and an
   entry under `llm_detector.entities`; register it in `KNOWN_ENTITY_TYPES`.
3. Add tests and a row to the README entity table.

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

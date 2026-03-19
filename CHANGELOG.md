# Changelog

All notable changes to `ai-guard` will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [0.1.0] — 2026-03-19

Initial release.

### Added

- **Hybrid detection engine** — regex + SpaCy NER + on-prem LLM (Ollama / OpenAI-compatible / Claude)
- **Regex detectors** — `CREDIT_CARD`, `EMAIL`, `PHONE`, `IBAN`, `TC_ID`, `IP_ADDRESS`, `ADDRESS`, `POSTAL_CODE`
- **NER detector** — `PERSON`, `ORG`, `ADDRESS` via SpaCy (English `en_core_web_sm` and Turkish `tr_core_news_*` models)
- **LLM detector** — `CUSTOM_SECRET` and any entity type via Ollama, OpenAI-compatible, or Anthropic Claude backends
- **Two actions** — `warn` (report, keep text) and `hash` (replace with `[TYPE:8hex]` using salted SHA-256)
- **Python API** — `LLMGuard`, `ScanResult`, `Violation`, `Action`; method-chaining configuration
- **YAML API** — declarative policy files with per-entity enable/action settings
- **`scan_batch()`** — fault-isolated batch scanning (one failure doesn't abort the batch)
- **CLI** — `ai-guard scan`, `ai-guard batch`, `ai-guard models list/setup/pull`
- **Environment variable overrides** — `LLMGUARD_SALT`, `LLMGUARD_LLM_URL`, `LLMGUARD_LLM_MODEL`, `LLMGUARD_LLM_API_KEY`, `LLMGUARD_LLM_TIMEOUT`, `LLMGUARD_SPACY_MODEL`
- **Config validation** — `validate_config()` checks actions, backend names, timeout values
- **SpaCy singleton cache** — thread-safe model cache prevents reloading 300-500 MB models per instance
- **Hallucination filtering** — structural validators per entity type (e.g. PERSON requires ≥2 words, TC_ID must be exactly 11 digits)
- **Overlap resolution** — longer span wins when two detectors produce overlapping matches
- **PEP 561 compliance** — `py.typed` marker for typed library consumers
- **Optional extras** — `[ner]` for SpaCy, `[claude]` for Anthropic SDK, `[all]` for everything
- **ReDoS protection** — adversarial input tests with 30-second timeout guardrails
- **Thread-safety** — concurrent scan tests validating shared-state safety

[0.1.0]: https://github.com/ai-guard/ai-guard/releases/tag/v0.1.0

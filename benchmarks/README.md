# Benchmarks

Head-to-head detection benchmark against
[Microsoft Presidio](https://github.com/microsoft/presidio). These scripts are not
part of the installed package and are not run in CI; the deterministic regression
checks live in `tests/benchmark/`.

| File | What it does |
|---|---|
| `compare.py` | Downloads the English corpus, runs an engine over a corpus, scores every engine that has predictions, and computes paired bootstrap intervals. |
| `corpus_tr.py` | 20 hand-written Turkish samples in the same schema as the English corpus. |
| `consistency.py` | Whether one Turkish name gets one value across grammatical case endings. |

## Corpora

- **English:** [presidio-research](https://github.com/microsoft/presidio-research)
  `data/synth_dataset_v2.json` (MIT), 1,500 synthetic samples built to Presidio's
  own taxonomy. `download` fetches it at commit `f3ff907` and checks its sha256.
- **Turkish:** 20 samples written for wardcat: names in lower case, local number
  formats, values inside prose. Every IBAN, TC ID and card passes its checksum.
  Because wardcat's side wrote it, treat Turkish results as direction, not proof.

## Scoring

A prediction is a true positive when it overlaps an unclaimed gold span whose type
maps to it; each gold span is claimed once. Predictions that claim nothing are false
positives, unclaimed gold spans are misses. Overlap is used rather than exact
boundaries so that multi-line addresses do not turn into a punctuation contest.

Only the eight types both engines offer are scored: `PERSON`, `ORGANIZATION`,
`CREDIT_CARD`, `PHONE_NUMBER`, `EMAIL_ADDRESS`, `IBAN_CODE`, `US_SSN`, `IP_ADDRESS`.
Turkish ID numbers are a wardcat-only type and are reported separately as coverage.

Both engines use the same spaCy model: `en_core_web_lg` for English (the model
Presidio's documentation recommends) and `tr_core_news_md` for Turkish. Neither is
tuned. Presidio runs its default `AnalyzerEngine`; for Turkish its card, IBAN,
e-mail, IP, SSN and phone recognizers are registered for the language, since
Presidio registers them for English only.

A wardcat scan that returns warnings (a missing or substituted NER model, an
unreachable LLM) stops the run instead of being scored, because it would measure a
different engine than the one named.

## Running it

wardcat and Presidio are installed in separate environments.

```bash
# wardcat, from the repository root
uv sync --extra ner
uv run python -m spacy download en_core_web_lg
uv run python -m spacy download tr_core_news_md

# Presidio, in its own environment
python -m venv .presidio && . .presidio/bin/activate
pip install presidio-analyzer
python -m spacy download en_core_web_lg
python -m spacy download tr_core_news_md
```

```bash
python benchmarks/compare.py download

# English: Presidio, then wardcat layer by layer
python benchmarks/compare.py predict --engine presidio        --corpus en   # Presidio env
uv run python benchmarks/compare.py predict --engine wardcat-regex   --corpus en  # regex only
uv run python benchmarks/compare.py predict --engine wardcat         --corpus en  # + NER
uv run python benchmarks/compare.py predict --engine wardcat-regions --corpus en  # + phone regions
uv run python benchmarks/compare.py score --corpus en

# English with the LLM layer: needs Ollama and `ollama pull qwen3:14b`.
# Seconds per sample, so the first 200; every engine is rescored on the same 200.
uv run python benchmarks/compare.py predict --engine wardcat-llm --corpus en --limit 200
uv run python benchmarks/compare.py score --corpus en --limit 200

# Turkish
python benchmarks/compare.py predict --engine presidio --corpus tr           # Presidio env
uv run python benchmarks/compare.py predict --engine wardcat     --corpus tr
uv run python benchmarks/compare.py predict --engine wardcat-llm --corpus tr
uv run python benchmarks/compare.py score --corpus tr

# Is a difference real?
uv run python benchmarks/compare.py bootstrap --corpus en --a wardcat-regions --b presidio

# Turkish case endings
uv run python benchmarks/consistency.py --engine wardcat --ner-model tr_core_news_lg
```

Predictions are written to `benchmarks/preds/` one line per sample, so an
interrupted run resumes where it stopped; scores go to `benchmarks/results/`. Both
directories and the downloaded corpus are git-ignored.

`score` and `bootstrap` only read predictions, so they run in either environment.
Delete a `preds/` file to rerun that engine from scratch.

## Results, 14 September 2026

wardcat 1.2.0 with unreleased fixes, presidio-analyzer 2.2.364, spaCy 3.8.16,
Apple M1 16 GB.

| English, 1,500 samples | Precision | Recall | F1 | Median latency |
|---|---|---|---|---|
| Presidio | 64.8% | 76.8% | 0.703 | 4 ms |
| wardcat, regex only | 100% | 16.9% | 0.290 | <1 ms |
| wardcat, regex + NER | 69.3% | 75.4% | 0.722 | 4 ms |
| wardcat, + phone regions | 69.7% | 77.9% | 0.736 | 4 ms |

On the first 200 samples, where the LLM layer was measured: Presidio 0.723,
regex + NER 0.750, + phone regions 0.768, + LLM 0.774 (2.8 s median per sample).
Turkish, 20 samples: Presidio 0.928, regex only 0.712, regex + NER 0.968,
+ LLM 0.989.

wardcat with phone regions minus Presidio: ΔF1 +0.033, 95% interval
[+0.026, +0.041]. With defaults wardcat finds fewer phone numbers (recall 21%
against 59%) and its overall recall is 1.4 points lower; its lead in that
configuration comes from precision. One synthetic corpus supports "better on this
benchmark", not "better everywhere".

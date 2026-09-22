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
| `corpus_hard.py` | 100 hard cases, 60 English and 40 Turkish, including decoys with no gold. |
| `sensitivity.py`, `corpus_sensitivity.py` | `is_sensitive()` on 100 texts labelled sensitive or not, against "anything found" from Presidio and wardcat regex + NER. |

## Corpora

- **English:** [presidio-research](https://github.com/microsoft/presidio-research)
  `data/synth_dataset_v2.json` (MIT), 1,500 synthetic samples built to Presidio's
  own taxonomy. `download` fetches it at commit `f3ff907` and checks its sha256.
- **Gretel (held out):**
  [gretelai/synthetic_pii_finance_multilingual](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual)
  (Apache-2.0), the English test split: 2,891 full-length financial documents
  (contracts, EDI messages, emails, logs) from a different generator and a
  different taxonomy. No wardcat rule was written while looking at it. `download`
  fetches the parquet at revision `7b844d1`, checks its sha256 and converts it to
  the schema above, which needs pyarrow. Its labels are incomplete: the second
  copy of an e-mail address in a `[x](mailto:x)` link and many phone numbers in
  letterheads are unlabelled, so every engine's precision reads low, by the same
  measure for each.
- **Turkish:** 20 samples written for wardcat: names in lower case, local number
  formats, values inside prose. Every IBAN, TC ID and card passes its checksum.
  Because wardcat's side wrote it, treat Turkish results as direction, not proof.
- **Hard cases and sensitivity:** also written by wardcat's side, and committed
  before any engine was run on them (`1dae1a9`); no case was changed after a
  result. The hard cases include wardcat's known weaknesses: unlabelled national
  phone numbers, Turkish sentence-initial run-ons, spoken and obfuscated values.

## Scoring

A prediction is a true positive when it overlaps an unclaimed gold span whose type
maps to it; each gold span is claimed once. Predictions that claim nothing are false
positives, unclaimed gold spans are misses. Overlap is used rather than exact
boundaries so that multi-line addresses do not turn into a punctuation contest.

Overlap cannot see a span that is too short: `Smith` for `John Smith` still counts.
The `hidden` column covers that — the share of gold characters of the scored types
that fall inside some prediction of any type, which is what redaction removes.

Only the eight types both engines offer are scored: `PERSON`, `ORGANIZATION`,
`CREDIT_CARD`, `PHONE_NUMBER`, `EMAIL_ADDRESS`, `IBAN_CODE`, `US_SSN`, `IP_ADDRESS`.
Turkish ID numbers are a wardcat-only type and are reported separately as coverage.

Both engines use the same spaCy model: `en_core_web_lg` for English (the model
Presidio's documentation recommends) and `tr_core_news_md` for Turkish. Neither
model is tuned. wardcat's span cleanup rules were written while reading errors on
the English (Presidio) corpus and on the hard cases; the Gretel corpus is held
out — no rule was written against it, and it is where a change is validated. Presidio runs its default `AnalyzerEngine`; for Turkish its card, IBAN,
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
uv pip install https://huggingface.co/turkish-nlp-suite/tr_core_news_md/resolve/main/tr_core_news_md-1.0-py3-none-any.whl

# Presidio, in its own environment
python -m venv .presidio && . .presidio/bin/activate
pip install presidio-analyzer
python -m spacy download en_core_web_lg
# The Turkish model is published by turkish-nlp-suite, not by spaCy, so
# `spacy download` cannot find it:
pip install https://huggingface.co/turkish-nlp-suite/tr_core_news_md/resolve/main/tr_core_news_md-1.0-py3-none-any.whl
```

```bash
uv run --with pyarrow python benchmarks/compare.py download

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

# Gretel: same engines, no LLM run (2,891 long documents)
python benchmarks/compare.py predict --engine presidio        --corpus gretel  # Presidio env
uv run python benchmarks/compare.py predict --engine wardcat-regex   --corpus gretel
uv run python benchmarks/compare.py predict --engine wardcat         --corpus gretel
uv run python benchmarks/compare.py predict --engine wardcat-regions --corpus gretel
uv run python benchmarks/compare.py score --corpus gretel

# Turkish
python benchmarks/compare.py predict --engine presidio --corpus tr           # Presidio env
uv run python benchmarks/compare.py predict --engine wardcat     --corpus tr
uv run python benchmarks/compare.py predict --engine wardcat-llm --corpus tr
uv run python benchmarks/compare.py score --corpus tr

# Is a difference real?
uv run python benchmarks/compare.py bootstrap --corpus en --a wardcat-regions --b presidio

# Hard cases: each sample runs through its own language's pipeline
python benchmarks/compare.py predict --engine presidio --corpus hard         # Presidio env
uv run python benchmarks/compare.py predict --engine wardcat     --corpus hard
uv run python benchmarks/compare.py predict --engine wardcat-llm --corpus hard
uv run python benchmarks/compare.py score --corpus hard

# is_sensitive()
python benchmarks/sensitivity.py predict --engine presidio                    # Presidio env
uv run python benchmarks/sensitivity.py predict --engine wardcat
uv run python benchmarks/sensitivity.py predict --engine wardcat-llm
uv run python benchmarks/sensitivity.py score

# Turkish case endings
uv run python benchmarks/consistency.py --engine wardcat --ner-model tr_core_news_lg
```

Predictions are written to `benchmarks/preds/` one line per sample, so an
interrupted run resumes where it stopped; scores go to `benchmarks/results/`. Both
directories and the downloaded corpus are git-ignored.

`score` and `bootstrap` only read predictions, so they run in either environment.
Delete a `preds/` file to rerun that engine from scratch.

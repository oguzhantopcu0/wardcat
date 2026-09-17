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

## Results, 17 September 2026

wardcat 1.2.0 with unreleased fixes, presidio-analyzer 2.2.364, spaCy 3.8.16,
Apple M1 16 GB. `hidden` is the share of gold PII characters removed.

| English, 1,500 samples | Precision | Recall | F1 | hidden | Median latency |
|---|---|---|---|---|---|
| Presidio | 64.8% | 76.8% | 0.703 | 85.3% | 4 ms |
| wardcat, regex only | 100% | 20.9% | 0.346 | 28.2% | <1 ms |
| wardcat, regex + NER | 72.6% | 79.4% | 0.759 | 87.7% | 4 ms |
| wardcat, + phone regions | 72.6% | 80.2% | 0.762 | 88.7% | 4 ms |

| Gretel, 2,891 documents | Precision | Recall | F1 | hidden | Median latency |
|---|---|---|---|---|---|
| Presidio | 33.5% | 74.1% | 0.462 | 77.5% | 75 ms |
| wardcat, regex only | 59.3% | 11.7% | 0.195 | 13.4% | 1 ms |
| wardcat, regex + NER | 38.7% | 72.1% | 0.504 | 74.7% | 46 ms |
| wardcat, + phone regions | 37.8% | 71.5% | 0.495 | 74.3% | 49 ms |

wardcat regex + NER minus Presidio, paired bootstrap:

| | ΔPrecision | ΔRecall | ΔF1 |
|---|---|---|---|
| English | +0.079 [+0.067, +0.091] | +0.026 [+0.013, +0.038] | +0.056 [+0.046, +0.066] |
| Gretel | +0.052 [+0.047, +0.057] | −0.021 [−0.026, −0.015] | +0.042 [+0.037, +0.047] |

On Gretel wardcat's lead is precision; Presidio finds more names and
organisations and removes more gold characters. The phone regions chosen for the
English corpus (`US GB BE ES FR DE`) lower phone precision on Gretel from 75% to
43%, which is the cost the documentation warns about.

The labelled-phone and NER span changes were written while reading errors on the
English corpus, so the English gain from them is partly fitted: regex + NER went
from 0.723 to 0.759 there, and from 0.497 to 0.504 on Gretel, which no rule was
written against.

On the first 200 English samples, where the LLM layer was measured: Presidio
0.723, regex + NER 0.776, + phone regions 0.785, + LLM (`qwen3:14b`) 0.785 at
1.9 s median per sample. Turkish, 20 samples: Presidio 0.928, regex only 0.712,
regex + NER 0.968, + LLM 0.989.

### Hard cases, 100 samples

| | Precision | Recall | F1 | hidden | Median latency |
|---|---|---|---|---|---|
| Presidio | 80.1% | 87.9% | 0.838 | 85.2% | 7 ms |
| wardcat, regex only | 96.4% | 42.7% | 0.592 | 48.5% | <1 ms |
| wardcat, regex + NER | 90.5% | 84.7% | 0.875 | 81.7% | 4 ms |
| wardcat, + LLM | 91.3% | 92.7% | 0.920 | 89.8% | 8.0 s |

Paired bootstrap against Presidio: regex + NER ΔF1 +0.037 [+0.001, +0.070],
recall −0.032 [−0.074, +0.000]; + LLM ΔF1 +0.082 [+0.046, +0.120]. English F1:
Presidio 0.862, regex + NER 0.889, + LLM 0.917. Turkish: 0.796, 0.851, 0.925.
Missed by every engine: spoken numbers, `[at]`/`(at)` e-mail obfuscation,
lower-case `deniz aydın`, `Trendyol`. Missed by wardcat without the LLM:
unlabelled national phone numbers and some Turkish names.

The first run of this set found four wardcat gaps, since fixed: Troy cards,
labelled SSNs without dashes, grouped TC numbers, and LLM card and IBAN
proposals that were not checked against their checksums. Before the fixes,
regex + NER scored 0.866 and + LLM 0.908.

### is_sensitive(), 100 texts

| | Accuracy | Precision | Recall | F1 | Median latency |
|---|---|---|---|---|---|
| Presidio, anything found | 63% | 64% | 60% | 0.619 | 7 ms |
| wardcat regex + NER, anything found | 72% | 70% | 76% | 0.731 | 4 ms |
| wardcat `is_sensitive()`, qwen3:14b | 88% | 82% | 98% | 0.891 | 2.0 s |

`is_sensitive()` missed one sensitive text (an unannounced acquisition, in German)
and flagged eleven harmless ones, six of them templates and format examples
(`name@example.com`, `XXX-XX-XXXX`) and two public customer-service numbers.
Detector-based answers miss what has no entity — diagnoses, layoffs, lawsuits,
unannounced deals — and flag public figures.

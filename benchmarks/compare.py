"""Compare wardcat with Microsoft Presidio on PII detection.

Predictions and scoring are separate steps, so each engine runs in its own
environment (Presidio and wardcat do not need to share one) and a slow LLM run can
be stopped and resumed:

    python benchmarks/compare.py download
    python benchmarks/compare.py predict --engine presidio        --corpus en
    python benchmarks/compare.py predict --engine wardcat         --corpus en
    python benchmarks/compare.py predict --engine wardcat-regions --corpus en
    python benchmarks/compare.py predict --engine wardcat-llm     --corpus en --limit 200
    python benchmarks/compare.py score     --corpus en
    python benchmarks/compare.py bootstrap --corpus en --a wardcat-regions --b presidio

Corpora
  en      presidio-research ``synth_dataset_v2.json`` (MIT), 1,500 samples: the
          competitor's own dataset and taxonomy. Downloaded at a pinned commit and
          checked against its sha256.
  gretel  gretelai/synthetic_pii_finance_multilingual (Apache-2.0), English test
          split, 2,891 full-length financial documents from a different generator
          and taxonomy. Neither engine's rules were written against it. Downloaded
          at a pinned revision, checked against its sha256 and converted to the
          same schema; the conversion needs pyarrow
          (``uv run --with pyarrow python benchmarks/compare.py download``).
  tr      20 hand-written Turkish samples (``corpus_tr.py``). Written by wardcat's
          side, which is home-field advantage; read those results as direction only.
  hard    100 hard cases, 60 English and 40 Turkish (``corpus_hard.py``), each run
          through the pipeline for its own language. Also written by wardcat's
          side, before any engine was run on it.

Scoring: a prediction is a true positive when it overlaps an unclaimed gold span
whose type maps to it; each gold span is claimed at most once. Predictions of a
scored type that claim nothing are false positives; unclaimed gold spans are
misses. Only the entity types both engines offer are scored.

Overlap scoring cannot see a span that is too short — "Smith" for "John Smith"
still counts. The ``hidden`` column closes that gap: the share of gold characters
of the scored types that fall inside some prediction, whatever its type, which is
what redaction actually removes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PREDS = HERE / "preds"
RESULTS = HERE / "results"

EN_FILE = DATA / "synth_dataset_v2.json"
EN_URL = (
    "https://raw.githubusercontent.com/microsoft/presidio-research/"
    "f3ff907eba57b8d380711ce7ca82a42696cd0490/data/synth_dataset_v2.json"
)
EN_SHA256 = "ec08a771ba8135314cafb60752b2295212222ba3a4cd75d73811839c699e0012"

GRETEL_PARQUET = DATA / "gretel_en_test.parquet"
GRETEL_FILE = DATA / "gretel_en_test.json"
GRETEL_URL = (
    "https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual/resolve/"
    "7b844d16738527a04264f50214cb426a4cea0897/data/English_test-00000-of-00001.parquet"
)
GRETEL_SHA256 = "c02b06d3c5b7c375525136d6f74acc52ab8fc07fa863d0aa50734910ec0ef2ad"
# Gretel label -> gold type. Labels not listed stay in the gold under their own
# name, so a prediction on them is neither credited nor matched.
GRETEL_LABELS = {
    "name": "PERSON",
    "first_name": "PERSON",
    "last_name": "PERSON",
    "company": "ORGANIZATION",
    "credit_card_number": "CREDIT_CARD",
    "phone_number": "PHONE_NUMBER",
    "email": "EMAIL_ADDRESS",
    "iban": "IBAN_CODE",
    "ssn": "US_SSN",
    "ipv4": "IP_ADDRESS",
    "ipv6": "IP_ADDRESS",
}
CORPORA = ("en", "gretel", "tr", "hard")

# Gold type -> (wardcat entity, Presidio entity). Only types both engines offer.
SCORED: dict[str, tuple[str, str]] = {
    "PERSON": ("PERSON", "PERSON"),
    "ORGANIZATION": ("ORG", "ORGANIZATION"),
    "CREDIT_CARD": ("CREDIT_CARD", "CREDIT_CARD"),
    "PHONE_NUMBER": ("PHONE", "PHONE_NUMBER"),
    "EMAIL_ADDRESS": ("EMAIL", "EMAIL_ADDRESS"),
    "IBAN_CODE": ("IBAN", "IBAN_CODE"),
    "US_SSN": ("SSN", "US_SSN"),
    # wardcat reports IPv6 as its own type; Presidio folds it into IP_ADDRESS.
    "IP_ADDRESS": ("IP_ADDRESS|IPv6", "IP_ADDRESS"),
}
# Gold types only wardcat offers: reported as coverage, never in the shared score.
WARDCAT_ONLY: dict[str, str] = {"TC_ID": "TC_ID"}

NER_MODEL = {"en": "en_core_web_lg", "gretel": "en_core_web_lg", "tr": "tr_core_news_md"}
LANGUAGE = {"en": "en", "gretel": "en", "tr": "tr"}
# The countries whose phone formats occur in the English corpus.
PHONE_REGIONS = ("US", "GB", "BE", "ES", "FR", "DE")
ENGINES = (
    "presidio",
    "wardcat-regex",
    "wardcat",
    "wardcat-regions",
    "wardcat-llm",
    "wardcat-entropy",
)
# Predicted types outside the shared score, counted per engine as a false-positive
# budget: nothing in any corpus is labelled as one, so every hit is a miss.
UNSCORED_BUDGET = ("HIGH_ENTROPY_STRING",)

Prediction = tuple[str, int, int]
Runner = Callable[[str], tuple[list[Prediction], list[str]]]


# ── corpora ───────────────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch(url: str, path: Path, sha256: str) -> None:
    if path.exists() and _sha256(path) == sha256:
        print(f"{path.name} already present and verified")
        return
    DATA.mkdir(exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - fixed URL
        path.write_bytes(response.read())
    digest = _sha256(path)
    if digest != sha256:
        path.unlink()
        sys.exit(f"sha256 mismatch for {url}: got {digest}, expected {sha256}")
    print(f"downloaded and verified {path}")


def _convert_gretel() -> None:
    """Rewrite the Gretel parquet in the synth_dataset_v2.json schema."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit(
            "converting the Gretel corpus needs pyarrow: "
            "uv run --with pyarrow python benchmarks/compare.py download"
        )
    out = []
    for row in pq.read_table(GRETEL_PARQUET).to_pylist():
        text = row["generated_text"]
        spans = [
            {
                "entity_type": GRETEL_LABELS.get(span["label"], span["label"]),
                "entity_value": text[span["start"] : span["end"]],
                "start_position": span["start"],
                "end_position": span["end"],
            }
            for span in json.loads(row["pii_spans"])
        ]
        out.append({"full_text": text, "spans": spans, "document_type": row["document_type"]})
    GRETEL_FILE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"converted {len(out)} documents to {GRETEL_FILE}")


def download(_: argparse.Namespace) -> None:
    _fetch(EN_URL, EN_FILE, EN_SHA256)
    _fetch(GRETEL_URL, GRETEL_PARQUET, GRETEL_SHA256)
    if not GRETEL_FILE.exists():
        _convert_gretel()


def load_corpus(name: str, limit: int = 0) -> list[dict]:
    if name in ("en", "gretel"):
        path = EN_FILE if name == "en" else GRETEL_FILE
        if not path.exists():
            sys.exit(f"{path.name} missing: run `python benchmarks/compare.py download` first")
        data = json.loads(path.read_text(encoding="utf-8"))
    elif name == "hard":
        sys.path.insert(0, str(HERE))
        from corpus_hard import as_dataset

        data = as_dataset()
    else:
        sys.path.insert(0, str(HERE))
        from corpus_tr import as_dataset

        data = as_dataset()
    return data[:limit] if limit else data


# ── engines ───────────────────────────────────────────────────────────────────


def presidio_engine(corpus: str) -> tuple[Runner, dict]:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    language = LANGUAGE[corpus]
    config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": language, "model_name": NER_MODEL[corpus]}],
        "ner_model_configuration": {
            "model_to_presidio_entity_mapping": {
                "PERSON": "PERSON",
                "PER": "PERSON",
                "ORG": "ORGANIZATION",
                "GPE": "LOCATION",
                "LOC": "LOCATION",
                "NORP": "NRP",
                "DATE": "DATE_TIME",
                "TIME": "DATE_TIME",
            },
            "low_confidence_score_multiplier": 0.4,
            "low_score_entity_names": ["ORG", "ORGANIZATION"],
        },
    }
    analyzer = AnalyzerEngine(
        nlp_engine=NlpEngineProvider(nlp_configuration=config).create_engine(),
        supported_languages=[language],
    )
    if language != "en":
        # Presidio registers its pattern recognizers for English only. Card, IBAN,
        # e-mail, IP and SSN formats do not depend on the language and a real
        # Turkish deployment would enable them; leaving them out would score a
        # configuration gap as a detection gap.
        from presidio_analyzer.predefined_recognizers import (
            CreditCardRecognizer,
            EmailRecognizer,
            IbanRecognizer,
            IpRecognizer,
            PhoneRecognizer,
            UsSsnRecognizer,
        )

        present = {r.name for r in analyzer.get_recognizers(language=language)}
        for recognizer in (
            CreditCardRecognizer(supported_language=language),
            EmailRecognizer(supported_language=language),
            IbanRecognizer(supported_language=language),
            IpRecognizer(supported_language=language),
            UsSsnRecognizer(supported_language=language),
            PhoneRecognizer(
                supported_language=language, supported_regions=["TR", "US", "UK", "DE"]
            ),
        ):
            if recognizer.name not in present:
                analyzer.registry.add_recognizer(recognizer)

    wanted = sorted({presidio for _, presidio in SCORED.values()})
    info = {"presidio_analyzer": version("presidio-analyzer"), "ner_model": NER_MODEL[corpus]}

    def run(text: str) -> tuple[list[Prediction], list[str]]:
        results = analyzer.analyze(text=text, language=language, entities=wanted)
        return [(r.entity_type, r.start, r.end) for r in results], []

    return run, info


def wardcat_engine(
    corpus: str,
    *,
    ner: bool = True,
    phone_regions: tuple[str, ...] = (),
    llm_model: str | None = None,
    entropy: bool = False,
) -> tuple[Runner, dict]:
    from wardcat import Action, Backend, Entity, Wardcat

    guard = Wardcat(salt="benchmark")
    if ner:
        guard = guard.with_ner(spacy_model=NER_MODEL[corpus], auto_download=False)
    if phone_regions:
        guard = guard.with_phone_regions(*phone_regions)
    regex = [
        Entity.CREDIT_CARD,
        Entity.PHONE,
        Entity.EMAIL,
        Entity.IBAN,
        Entity.SSN,
        Entity.IP_ADDRESS,
        Entity.IPv6,
        Entity.TC_ID,
    ]
    layers: dict[Entity, list[str]] = {entity: ["regex"] for entity in regex}
    if ner:
        layers[Entity.PERSON] = ["ner"]
        layers[Entity.ORG] = ["ner"]
    if llm_model:
        guard = guard.with_llm(backend=Backend.OLLAMA, model=llm_model, timeout=600)
        guard.remove_entity(Entity.ALL)  # drop the LLM layer's own default policy
        layers = {entity: [*lyr, "llm"] for entity, lyr in layers.items()}
    for entity, lyr in layers.items():
        guard.add_entity(entity, Action.REDACT, layers=lyr)
    if entropy:
        # Opt-in and under the floor by design; the benchmark enables it the way
        # a user would, to count what it would flag.
        guard.add_entity(Entity.HIGH_ENTROPY_STRING, Action.REDACT, min_confidence=0.7)

    info = {
        "wardcat": version("wardcat"),
        "ner_model": NER_MODEL[corpus] if ner else None,
        "phone_regions": list(phone_regions),
        "llm": {"backend": "ollama", "model": llm_model} if llm_model else None,
        "entropy": entropy,
    }

    def run(text: str) -> tuple[list[Prediction], list[str]]:
        result = guard.scan(text)
        # A fourth element, the layer that found the span, so the score can be
        # broken down by layer. Presidio predictions stay three-tuples.
        preds = [(v.entity_type, v.start, v.end, v.source) for v in result.violations]
        return preds, list(result.warnings)  # type: ignore[return-value]

    return run, info


def build_engine(engine: str, corpus: str, llm_model: str) -> tuple[Runner, dict]:
    if engine == "presidio":
        return presidio_engine(corpus)
    if engine == "wardcat-regex":
        return wardcat_engine(corpus, ner=False)
    if engine == "wardcat-regions":
        return wardcat_engine(corpus, phone_regions=PHONE_REGIONS)
    if engine == "wardcat-llm":
        return wardcat_engine(corpus, llm_model=llm_model)
    if engine == "wardcat-entropy":
        return wardcat_engine(corpus, entropy=True)
    return wardcat_engine(corpus)


# ── predict ───────────────────────────────────────────────────────────────────


def predict(args: argparse.Namespace) -> None:
    corpus = load_corpus(args.corpus, args.limit)
    PREDS.mkdir(exist_ok=True)
    out = PREDS / f"{args.corpus}-{args.engine}.jsonl"
    done: set[int] = set()
    if out.exists():
        done = {json.loads(line)["i"] for line in out.read_text().splitlines() if line.strip()}
    todo = [i for i in range(len(corpus)) if i not in done]
    print(f"{args.engine}/{args.corpus}: {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return

    # A mixed corpus names each sample's language; build one engine per language.
    runners: dict[str, Runner] = {}
    infos: dict[str, dict] = {}
    for language in sorted({corpus[i].get("language", args.corpus) for i in todo}):
        runners[language], infos[language] = build_engine(args.engine, language, args.llm_model)
    info = infos[args.corpus] if list(infos) == [args.corpus] else infos
    (PREDS / f"{args.corpus}-{args.engine}.info.json").write_text(json.dumps(info, indent=2))

    with out.open("a") as fh:
        for n, i in enumerate(todo, 1):
            run = runners[corpus[i].get("language", args.corpus)]
            started = time.perf_counter()
            preds, warnings = run(corpus[i]["full_text"])
            ms = (time.perf_counter() - started) * 1000
            if warnings and not args.allow_warnings:
                # A degraded scan measures a different engine than the one named:
                # a missing NER model, a substituted one, an unreachable LLM.
                sys.exit(f"sample {i} scanned with warnings, refusing to score it: {warnings}")
            fh.write(json.dumps({"i": i, "ms": ms, "pred": preds, "warnings": warnings}) + "\n")
            fh.flush()
            if n % args.every == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  last {ms:.0f} ms", flush=True)


# ── score ─────────────────────────────────────────────────────────────────────


def _tally() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0}


def _prf(t: dict[str, int]) -> tuple[float, float, float]:
    p = t["tp"] / (t["tp"] + t["fp"]) if t["tp"] + t["fp"] else 0.0
    r = t["tp"] / (t["tp"] + t["fn"]) if t["tp"] + t["fn"] else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _match(gold: list[dict], preds: list[Prediction], wanted: str, tally: dict[str, int]) -> None:
    """``wanted`` is one entity type, or several joined with ``|``."""
    types = set(wanted.split("|"))
    claimed: set[int] = set()
    for pred in (p for p in preds if p[0] in types):
        start, end = pred[1], pred[2]
        source = pred[3] if len(pred) > 3 else ""
        for k, span in enumerate(gold):
            if k not in claimed and start < span["end_position"] and span["start_position"] < end:
                claimed.add(k)
                tally["tp"] += 1
                _count_source(tally, source, "tp")
                break
        else:
            tally["fp"] += 1
            _count_source(tally, source, "fp")
    tally["fn"] += len(gold) - len(claimed)


def _count_source(tally: dict, source: str, key: str) -> None:
    """Per-layer TP/FP inside a tally, kept under a nested key."""
    if not source:
        return
    per = tally.setdefault("by_source", {})
    per.setdefault(source, {"tp": 0, "fp": 0})[key] += 1


def _load_preds(corpus: str, engine: str, n: int) -> dict[int, dict]:
    path = PREDS / f"{corpus}-{engine}.jsonl"
    rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {row["i"]: row for row in rows if row["i"] < n}


def _hidden_chars(sample: dict, preds: list[Prediction]) -> tuple[int, int]:
    """Gold characters of the scored types, and how many of them some prediction covers."""
    covered = set()
    for pred in preds:
        covered.update(range(pred[1], pred[2]))
    total = hidden = 0
    for span in sample["spans"]:
        if span["entity_type"] in SCORED:
            chars = range(span["start_position"], span["end_position"])
            total += len(chars)
            hidden += sum(1 for c in chars if c in covered)
    return total, hidden


def _sample_tallies(sample: dict, preds: list[Prediction], engine: str) -> dict[str, dict]:
    per_type = {}
    for gold_type, (wardcat_type, presidio_type) in SCORED.items():
        wanted = wardcat_type if engine.startswith("wardcat") else presidio_type
        gold = [s for s in sample["spans"] if s["entity_type"] == gold_type]
        per_type[gold_type] = _tally()
        _match(gold, preds, wanted, per_type[gold_type])
    return per_type


def score(args: argparse.Namespace) -> None:
    corpus = load_corpus(args.corpus, args.limit)
    n = len(corpus)
    engines = [e for e in ENGINES if (PREDS / f"{args.corpus}-{e}.jsonl").exists()]
    table = {}
    for engine in engines:
        rows = _load_preds(args.corpus, engine, n)
        if len(rows) < n:
            print(f"{engine}: {len(rows)}/{n} samples predicted, skipped (use --limit)")
            continue
        per_type: dict[str, dict] = defaultdict(_tally)
        coverage: dict[str, dict] = defaultdict(_tally)
        gold_chars = hidden_chars = 0
        for i, sample in enumerate(corpus):
            preds = [tuple(p) for p in rows[i]["pred"]]
            total, hidden = _hidden_chars(sample, preds)
            gold_chars += total
            hidden_chars += hidden
            for gold_type, t in _sample_tallies(sample, preds, engine).items():
                for key in ("tp", "fp", "fn"):
                    per_type[gold_type][key] += t[key]
                for source, counts in t.get("by_source", {}).items():
                    acc = per_type[gold_type].setdefault("by_source", {})
                    acc.setdefault(source, {"tp": 0, "fp": 0})
                    acc[source]["tp"] += counts["tp"]
                    acc[source]["fp"] += counts["fp"]
            if engine.startswith("wardcat"):
                for gold_type, wanted in WARDCAT_ONLY.items():
                    gold = [s for s in sample["spans"] if s["entity_type"] == gold_type]
                    if gold:
                        _match(gold, preds, wanted, coverage[gold_type])
        unscored = {
            t: sum(1 for i in rows for p in rows[i]["pred"] if p[0] == t) for t in UNSCORED_BUDGET
        }
        micro = _tally()
        by_source: dict[str, dict[str, int]] = {}
        for t in per_type.values():
            for key in micro:
                micro[key] += t[key]
            for source, counts in t.get("by_source", {}).items():
                acc = by_source.setdefault(source, {"tp": 0, "fp": 0})
                acc["tp"] += counts["tp"]
                acc["fp"] += counts["fp"]
        latencies = sorted(row["ms"] for row in rows.values())
        table[engine] = {
            "micro": micro,
            "by_source": by_source,
            "unscored": unscored,
            "micro_prf": _prf(micro),
            "hidden": hidden_chars / gold_chars if gold_chars else 0.0,
            "per_type": {k: {**v, "prf": _prf(v)} for k, v in per_type.items()},
            "coverage": {k: {**v, "prf": _prf(v)} for k, v in coverage.items()},
            "latency_ms": {
                "median": statistics.median(latencies),
                "p95": latencies[int(0.95 * (len(latencies) - 1))],
            },
            "info": json.loads((PREDS / f"{args.corpus}-{engine}.info.json").read_text()),
        }

    print(f"\n{args.corpus} corpus, {n} samples, micro-average over {len(SCORED)} shared types")
    print(
        f"{'engine':<17}{'P':>8}{'R':>8}{'F1':>8}{'hidden':>8}{'TP/FP/FN':>18}"
        f"{'median':>10}{'p95':>10}"
    )
    for engine, r in table.items():
        p, rc, f = r["micro_prf"]
        m = r["micro"]
        counts = f"{m['tp']}/{m['fp']}/{m['fn']}"
        lat = r["latency_ms"]
        print(
            f"{engine:<17}{p:>8.1%}{rc:>8.1%}{f:>8.3f}{r['hidden']:>8.1%}{counts:>18}"
            f"{lat['median']:>8.0f}ms{lat['p95']:>8.0f}ms"
        )
    print(f"\n{'F1 by type':<15}{'gold':>6}" + "".join(f"{e:>17}" for e in table))
    for gold_type in SCORED:
        first = next(iter(table.values()))["per_type"][gold_type]
        row = f"{gold_type:<15}{first['tp'] + first['fn']:>6}"
        row += "".join(f"{r['per_type'][gold_type]['prf'][2]:>17.3f}" for r in table.values())
        print(row)
    for engine, r in table.items():
        for gold_type, t in r["coverage"].items():
            print(f"coverage {engine} {gold_type}: {t['tp']}/{t['tp'] + t['fn']} found")
    for engine, r in table.items():
        for t, n in r["unscored"].items():
            if n or "entropy" in engine:
                print(
                    f"unscored {engine} {t}: {n} prediction(s) of a type no corpus labels — "
                    "false positives, unless the corpus labels them under another name"
                )
    for engine, r in table.items():
        if r["by_source"]:
            parts = ", ".join(
                f"{src} TP {c['tp']} FP {c['fp']}" for src, c in sorted(r["by_source"].items())
            )
            print(f"by layer {engine}: {parts}")
    if "category" in corpus[0]:
        _print_groups(corpus, table, args, "language")
        _print_groups(corpus, table, args, "category")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.corpus}-{args.limit or 'all'}.json"
    out.write_text(json.dumps(table, indent=2))
    print(f"\nwrote {out}")


def _print_groups(corpus: list[dict], table: dict, args: argparse.Namespace, key: str) -> None:
    """F1 and false positives per value of ``key``; decoy groups have no gold, so FP only."""
    groups = sorted({sample[key] for sample in corpus})
    print(f"\n{key + ': F1 (FP)':<22}" + "".join(f"{e:>17}" for e in table))
    for group in groups:
        cells = ""
        for engine in table:
            rows = _load_preds(args.corpus, engine, len(corpus))
            micro = _tally()
            for i, sample in enumerate(corpus):
                if sample[key] != group:
                    continue
                preds = [tuple(p) for p in rows[i]["pred"]]
                for t in _sample_tallies(sample, preds, engine).values():
                    for k in micro:
                        micro[k] += t[k]
                table[engine].setdefault(key, {})[group] = {**micro, "prf": _prf(micro)}
            f1 = f"{_prf(micro)[2]:.3f}" if micro["tp"] + micro["fn"] else "  -  "
            cells += f"{f1 + ' (' + str(micro['fp']) + ')':>17}"
        print(f"{group:<22}{cells}")


# ── bootstrap ─────────────────────────────────────────────────────────────────


def bootstrap(args: argparse.Namespace) -> None:
    """Paired bootstrap interval for the F1, precision and recall differences a − b."""
    corpus = load_corpus(args.corpus, args.limit)
    n = len(corpus)

    def sample_counts(engine: str) -> list[tuple[int, int, int]]:
        rows = _load_preds(args.corpus, engine, n)
        counts = []
        for i, sample in enumerate(corpus):
            tallies = _sample_tallies(sample, [tuple(p) for p in rows[i]["pred"]], engine)
            counts.append(tuple(sum(t[k] for t in tallies.values()) for k in ("tp", "fp", "fn")))
        return counts

    a, b = sample_counts(args.a), sample_counts(args.b)

    def prf(counts: list[tuple[int, int, int]], idx: list[int]) -> tuple[float, float, float]:
        tp, fp, fn = (sum(counts[i][k] for i in idx) for k in range(3))
        return _prf({"tp": tp, "fp": fp, "fn": fn})

    everything = list(range(n))
    point = [x - y for x, y in zip(prf(a, everything), prf(b, everything), strict=True)]
    rng = random.Random(args.seed)
    draws: list[list[float]] = [[], [], []]
    for _ in range(args.reps):
        idx = [rng.randrange(n) for _ in range(n)]
        for k, (x, y) in enumerate(zip(prf(a, idx), prf(b, idx), strict=True)):
            draws[k].append(x - y)
    lo, hi = int(0.025 * args.reps), int(0.975 * args.reps) - 1
    print(f"{args.corpus}, {n} samples, {args.reps} paired resamples: {args.a} − {args.b}")
    for label, k in (("precision", 0), ("recall", 1), ("F1", 2)):
        ordered = sorted(draws[k])
        print(f"  Δ{label:<10}{point[k]:+.3f}   95% [{ordered[lo]:+.3f}, {ordered[hi]:+.3f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("download", help="fetch and verify the English corpus")

    p = sub.add_parser("predict", help="run one engine over a corpus")
    p.add_argument("--engine", required=True, choices=ENGINES)
    p.add_argument("--corpus", required=True, choices=CORPORA)
    p.add_argument("--limit", type=int, default=0, help="only the first N samples")
    p.add_argument("--llm-model", default="qwen3:14b", help="Ollama model for wardcat-llm")
    p.add_argument("--every", type=int, default=100, help="progress interval")
    p.add_argument("--allow-warnings", action="store_true", help="score degraded scans anyway")

    s = sub.add_parser("score", help="score every engine that has predictions")
    s.add_argument("--corpus", required=True, choices=CORPORA)
    s.add_argument("--limit", type=int, default=0)

    bs = sub.add_parser("bootstrap", help="confidence interval for a difference")
    bs.add_argument("--corpus", required=True, choices=CORPORA)
    bs.add_argument("--limit", type=int, default=0)
    bs.add_argument("--a", required=True, choices=ENGINES)
    bs.add_argument("--b", required=True, choices=ENGINES)
    bs.add_argument("--reps", type=int, default=4000)
    bs.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()
    {"download": download, "predict": predict, "score": score, "bootstrap": bootstrap}[args.cmd](
        args
    )


if __name__ == "__main__":
    main()

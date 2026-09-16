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
  en  presidio-research ``synth_dataset_v2.json`` (MIT), 1,500 samples: the
      competitor's own dataset and taxonomy. Downloaded at a pinned commit and
      checked against its sha256.
  tr  20 hand-written Turkish samples (``corpus_tr.py``). Written by wardcat's
      side, which is home-field advantage; read those results as direction only.

Scoring: a prediction is a true positive when it overlaps an unclaimed gold span
whose type maps to it; each gold span is claimed at most once. Predictions of a
scored type that claim nothing are false positives; unclaimed gold spans are
misses. Only the entity types both engines offer are scored.
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

# Gold type -> (wardcat entity, Presidio entity). Only types both engines offer.
SCORED: dict[str, tuple[str, str]] = {
    "PERSON": ("PERSON", "PERSON"),
    "ORGANIZATION": ("ORG", "ORGANIZATION"),
    "CREDIT_CARD": ("CREDIT_CARD", "CREDIT_CARD"),
    "PHONE_NUMBER": ("PHONE", "PHONE_NUMBER"),
    "EMAIL_ADDRESS": ("EMAIL", "EMAIL_ADDRESS"),
    "IBAN_CODE": ("IBAN", "IBAN_CODE"),
    "US_SSN": ("SSN", "US_SSN"),
    "IP_ADDRESS": ("IP_ADDRESS", "IP_ADDRESS"),
}
# Gold types only wardcat offers: reported as coverage, never in the shared score.
WARDCAT_ONLY: dict[str, str] = {"TC_ID": "TC_ID"}

NER_MODEL = {"en": "en_core_web_lg", "tr": "tr_core_news_md"}
# The countries whose phone formats occur in the English corpus.
PHONE_REGIONS = ("US", "GB", "BE", "ES", "FR", "DE")
ENGINES = ("presidio", "wardcat-regex", "wardcat", "wardcat-regions", "wardcat-llm")

Prediction = tuple[str, int, int]
Runner = Callable[[str], tuple[list[Prediction], list[str]]]


# ── corpora ───────────────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(_: argparse.Namespace) -> None:
    if EN_FILE.exists() and _sha256(EN_FILE) == EN_SHA256:
        print(f"{EN_FILE.name} already present and verified")
        return
    DATA.mkdir(exist_ok=True)
    with urllib.request.urlopen(EN_URL, timeout=60) as response:  # noqa: S310 - fixed URL
        EN_FILE.write_bytes(response.read())
    digest = _sha256(EN_FILE)
    if digest != EN_SHA256:
        EN_FILE.unlink()
        sys.exit(f"sha256 mismatch for {EN_URL}: got {digest}, expected {EN_SHA256}")
    print(f"downloaded and verified {EN_FILE}")


def load_corpus(name: str, limit: int = 0) -> list[dict]:
    if name == "en":
        if not EN_FILE.exists():
            sys.exit("English corpus missing: run `python benchmarks/compare.py download` first")
        data = json.loads(EN_FILE.read_text(encoding="utf-8"))
    else:
        sys.path.insert(0, str(HERE))
        from corpus_tr import as_dataset

        data = as_dataset()
    return data[:limit] if limit else data


# ── engines ───────────────────────────────────────────────────────────────────


def presidio_engine(corpus: str) -> tuple[Runner, dict]:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": corpus, "model_name": NER_MODEL[corpus]}],
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
        supported_languages=[corpus],
    )
    if corpus != "en":
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

        present = {r.name for r in analyzer.get_recognizers(language=corpus)}
        for recognizer in (
            CreditCardRecognizer(supported_language=corpus),
            EmailRecognizer(supported_language=corpus),
            IbanRecognizer(supported_language=corpus),
            IpRecognizer(supported_language=corpus),
            UsSsnRecognizer(supported_language=corpus),
            PhoneRecognizer(supported_language=corpus, supported_regions=["TR", "US", "UK", "DE"]),
        ):
            if recognizer.name not in present:
                analyzer.registry.add_recognizer(recognizer)

    wanted = sorted({presidio for _, presidio in SCORED.values()})
    info = {"presidio_analyzer": version("presidio-analyzer"), "ner_model": NER_MODEL[corpus]}

    def run(text: str) -> tuple[list[Prediction], list[str]]:
        results = analyzer.analyze(text=text, language=corpus, entities=wanted)
        return [(r.entity_type, r.start, r.end) for r in results], []

    return run, info


def wardcat_engine(
    corpus: str,
    *,
    ner: bool = True,
    phone_regions: tuple[str, ...] = (),
    llm_model: str | None = None,
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

    info = {
        "wardcat": version("wardcat"),
        "ner_model": NER_MODEL[corpus] if ner else None,
        "phone_regions": list(phone_regions),
        "llm": {"backend": "ollama", "model": llm_model} if llm_model else None,
    }

    def run(text: str) -> tuple[list[Prediction], list[str]]:
        result = guard.scan(text)
        return [(v.entity_type, v.start, v.end) for v in result.violations], list(result.warnings)

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

    run, info = build_engine(args.engine, args.corpus, args.llm_model)
    (PREDS / f"{args.corpus}-{args.engine}.info.json").write_text(json.dumps(info, indent=2))

    with out.open("a") as fh:
        for n, i in enumerate(todo, 1):
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
    claimed: set[int] = set()
    for _, start, end in (p for p in preds if p[0] == wanted):
        for k, span in enumerate(gold):
            if k not in claimed and start < span["end_position"] and span["start_position"] < end:
                claimed.add(k)
                tally["tp"] += 1
                break
        else:
            tally["fp"] += 1
    tally["fn"] += len(gold) - len(claimed)


def _load_preds(corpus: str, engine: str, n: int) -> dict[int, dict]:
    path = PREDS / f"{corpus}-{engine}.jsonl"
    rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {row["i"]: row for row in rows if row["i"] < n}


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
        for i, sample in enumerate(corpus):
            preds = [tuple(p) for p in rows[i]["pred"]]
            for gold_type, t in _sample_tallies(sample, preds, engine).items():
                for key in t:
                    per_type[gold_type][key] += t[key]
            if engine.startswith("wardcat"):
                for gold_type, wanted in WARDCAT_ONLY.items():
                    gold = [s for s in sample["spans"] if s["entity_type"] == gold_type]
                    if gold:
                        _match(gold, preds, wanted, coverage[gold_type])
        micro = _tally()
        for t in per_type.values():
            for key in micro:
                micro[key] += t[key]
        latencies = sorted(row["ms"] for row in rows.values())
        table[engine] = {
            "micro": micro,
            "micro_prf": _prf(micro),
            "per_type": {k: {**v, "prf": _prf(v)} for k, v in per_type.items()},
            "coverage": {k: {**v, "prf": _prf(v)} for k, v in coverage.items()},
            "latency_ms": {
                "median": statistics.median(latencies),
                "p95": latencies[int(0.95 * (len(latencies) - 1))],
            },
            "info": json.loads((PREDS / f"{args.corpus}-{engine}.info.json").read_text()),
        }

    print(f"\n{args.corpus} corpus, {n} samples, micro-average over {len(SCORED)} shared types")
    print(f"{'engine':<17}{'P':>8}{'R':>8}{'F1':>8}{'TP/FP/FN':>18}{'median':>10}{'p95':>10}")
    for engine, r in table.items():
        p, rc, f = r["micro_prf"]
        m = r["micro"]
        counts = f"{m['tp']}/{m['fp']}/{m['fn']}"
        lat = r["latency_ms"]
        print(
            f"{engine:<17}{p:>8.1%}{rc:>8.1%}{f:>8.3f}{counts:>18}"
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

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.corpus}-{args.limit or 'all'}.json"
    out.write_text(json.dumps(table, indent=2))
    print(f"\nwrote {out}")


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
    p.add_argument("--corpus", required=True, choices=["en", "tr"])
    p.add_argument("--limit", type=int, default=0, help="only the first N samples")
    p.add_argument("--llm-model", default="qwen3:14b", help="Ollama model for wardcat-llm")
    p.add_argument("--every", type=int, default=100, help="progress interval")
    p.add_argument("--allow-warnings", action="store_true", help="score degraded scans anyway")

    s = sub.add_parser("score", help="score every engine that has predictions")
    s.add_argument("--corpus", required=True, choices=["en", "tr"])
    s.add_argument("--limit", type=int, default=0)

    bs = sub.add_parser("bootstrap", help="confidence interval for a difference")
    bs.add_argument("--corpus", required=True, choices=["en", "tr"])
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

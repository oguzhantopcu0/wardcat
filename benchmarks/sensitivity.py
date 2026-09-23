"""Is this text sensitive? ``Wardcat.is_sensitive()`` against detector-based answers.

``is_sensitive()`` asks the LLM for one holistic decision. Neither Presidio nor
wardcat's regex and NER layers make that decision, so the only way to put them to
the same use is to call a text sensitive when they find anything in it. That is
how they are scored here, with the entity types that say nothing on their own
left out on both sides: organisations, locations, nationalities, and for Presidio
every date and URL.

    python benchmarks/sensitivity.py predict --engine presidio      # Presidio env
    uv run python benchmarks/sensitivity.py predict --engine wardcat
    uv run python benchmarks/sensitivity.py predict --engine wardcat-llm
    uv run python benchmarks/sensitivity.py score

No German or French spaCy model is used: those texts run through the English
pipelines on both sides.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from corpus_sensitivity import SAMPLES  # noqa: E402

PREDS = HERE / "preds"
RESULTS = HERE / "results"
ENGINES = ("presidio", "wardcat", "wardcat-llm", "wardcat-classify")
# Corpus category → the category name classify() uses.
GOLD_TO_VERDICT = {"special": "special_category", "business": "business_confidential"}
NER_MODEL = {"en": "en_core_web_lg", "tr": "tr_core_news_md"}
PIPELINE = {"en": "en", "tr": "tr", "de": "en", "fr": "en"}
PRESIDIO_IGNORED = {"ORGANIZATION", "LOCATION", "NRP", "DATE_TIME", "URL"}


def presidio_classifier():
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer.predefined_recognizers import (
        CreditCardRecognizer,
        EmailRecognizer,
        IbanRecognizer,
        IpRecognizer,
        PhoneRecognizer,
    )

    config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": code, "model_name": model} for code, model in NER_MODEL.items()],
    }
    analyzer = AnalyzerEngine(
        nlp_engine=NlpEngineProvider(nlp_configuration=config).create_engine(),
        supported_languages=list(NER_MODEL),
    )
    # Presidio registers its pattern recognizers for English only; a Turkish
    # deployment would add the language-independent ones.
    for recognizer in (
        CreditCardRecognizer(supported_language="tr"),
        EmailRecognizer(supported_language="tr"),
        IbanRecognizer(supported_language="tr"),
        IpRecognizer(supported_language="tr"),
        PhoneRecognizer(supported_language="tr", supported_regions=["TR", "US", "UK", "DE"]),
    ):
        analyzer.registry.add_recognizer(recognizer)

    def classify(text: str, language: str) -> tuple[bool, list]:
        found = [
            r
            for r in analyzer.analyze(text=text, language=PIPELINE[language])
            if r.entity_type not in PRESIDIO_IGNORED
        ]
        return bool(found), sorted({r.entity_type for r in found})

    return classify, {"presidio_analyzer": version("presidio-analyzer")}


def wardcat_classifier():
    from wardcat import Action, Entity, Wardcat

    skipped = {Entity.ALL, Entity.ORG, Entity.LOCATION, Entity.NRP}
    guards = {}
    for code, model in NER_MODEL.items():
        guard = Wardcat(salt="benchmark").with_ner(spacy_model=model, auto_download=False)
        for entity in Entity:
            if entity not in skipped:
                guard.add_entity(entity, Action.REDACT)
        guards[code] = guard

    def classify(text: str, language: str) -> tuple[bool, list]:
        result = guards[PIPELINE[language]].scan(text)
        if result.warnings:
            sys.exit(f"scan returned warnings, refusing to score it: {result.warnings}")
        return bool(result.violations), sorted({v.entity_type for v in result.violations})

    return classify, {"wardcat": version("wardcat"), "ner_models": NER_MODEL}


def wardcat_llm_classifier(model: str):
    from wardcat import Backend, Wardcat

    guard = Wardcat(salt="benchmark").with_llm(backend=Backend.OLLAMA, model=model, timeout=600)

    def classify(text: str, language: str) -> tuple[bool, list]:
        return guard.is_sensitive(text), []

    return classify, {"wardcat": version("wardcat"), "llm": {"backend": "ollama", "model": model}}


def wardcat_classify_classifier(model: str):
    """classify(): the boolean is scored like the others, the categories on their own."""
    from wardcat import Backend, Wardcat

    guard = Wardcat(salt="benchmark").with_llm(backend=Backend.OLLAMA, model=model, timeout=600)

    def classify(text: str, language: str) -> tuple[bool, list]:
        verdict = guard.classify(text)
        return verdict.sensitive, list(verdict.categories)

    return classify, {"wardcat": version("wardcat"), "llm": {"backend": "ollama", "model": model}}


def predict(args: argparse.Namespace) -> None:
    PREDS.mkdir(exist_ok=True)
    out = PREDS / f"sens-{args.engine}.jsonl"
    done: set[int] = set()
    if out.exists():
        done = {json.loads(line)["i"] for line in out.read_text().splitlines() if line.strip()}
    todo = [i for i in range(len(SAMPLES)) if i not in done]
    print(f"{args.engine}: {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return
    if args.engine == "presidio":
        classify, info = presidio_classifier()
    elif args.engine == "wardcat":
        classify, info = wardcat_classifier()
    elif args.engine == "wardcat-classify":
        classify, info = wardcat_classify_classifier(args.llm_model)
    else:
        classify, info = wardcat_llm_classifier(args.llm_model)
    (PREDS / f"sens-{args.engine}.info.json").write_text(json.dumps(info, indent=2))
    with out.open("a") as fh:
        for n, i in enumerate(todo, 1):
            _, _, language, text = SAMPLES[i]
            started = time.perf_counter()
            sensitive, why = classify(text, language)
            ms = (time.perf_counter() - started) * 1000
            fh.write(json.dumps({"i": i, "ms": ms, "sensitive": sensitive, "why": why}) + "\n")
            fh.flush()
            if n % 20 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  last {ms:.0f} ms", flush=True)


def _metrics(pairs: list[tuple[bool, bool]]) -> dict:
    tp = sum(1 for gold, pred in pairs if gold and pred)
    fp = sum(1 for gold, pred in pairs if not gold and pred)
    fn = sum(1 for gold, pred in pairs if gold and not pred)
    tn = len(pairs) - tp - fp - fn
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / len(pairs),
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
    }


def score(_: argparse.Namespace) -> None:
    table = {}
    for engine in ENGINES:
        path = PREDS / f"sens-{engine}.jsonl"
        if not path.exists():
            continue
        rows = {json.loads(line)["i"]: json.loads(line) for line in path.read_text().splitlines()}
        if len(rows) < len(SAMPLES):
            print(f"{engine}: {len(rows)}/{len(SAMPLES)} predicted, skipped")
            continue
        pairs = [(SAMPLES[i][0], rows[i]["sensitive"]) for i in range(len(SAMPLES))]
        by_language: dict[str, list] = defaultdict(list)
        by_category: dict[str, list] = defaultdict(list)
        for i, (label, category, language, _) in enumerate(SAMPLES):
            by_language[language].append(pairs[i])
            by_category[f"{'sensitive' if label else 'harmless'}/{category}"].append(pairs[i])
        # Category accuracy, for engines that name categories: a sensitive text
        # is right when its gold category is among the ones the verdict names.
        category_hits = [
            GOLD_TO_VERDICT.get(cat, cat) in rows[i]["why"]
            for i, (label, cat, _, _) in enumerate(SAMPLES)
            if label and rows[i]["sensitive"]
        ]
        table[engine] = {
            "category_accuracy": (
                sum(category_hits) / len(category_hits)
                if engine == "wardcat-classify" and category_hits
                else None
            ),
            "all": _metrics(pairs),
            "language": {k: _metrics(v) for k, v in by_language.items()},
            "category": {
                k: sum(g == p for g, p in v) / len(v) for k, v in sorted(by_category.items())
            },
            "median_ms": statistics.median(row["ms"] for row in rows.values()),
            "errors": [
                {"i": i, "gold": SAMPLES[i][0], "text": SAMPLES[i][3], "why": rows[i]["why"]}
                for i in range(len(SAMPLES))
                if SAMPLES[i][0] != rows[i]["sensitive"]
            ],
        }

    print(f"\n{len(SAMPLES)} texts, 50 sensitive and 50 harmless")
    print(f"{'engine':<13}{'acc':>7}{'P':>7}{'R':>7}{'F1':>7}{'TP/FP/FN/TN':>16}{'median':>10}")
    for engine, r in table.items():
        m = r["all"]
        counts = f"{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}"
        print(
            f"{engine:<13}{m['accuracy']:>7.0%}{m['precision']:>7.0%}{m['recall']:>7.0%}"
            f"{m['f1']:>7.3f}{counts:>16}{r['median_ms']:>8.0f}ms"
        )
    for engine, r in table.items():
        if r["category_accuracy"] is not None:
            print(f"category accuracy {engine}: {r['category_accuracy']:.0%} of true positives")
    print(f"\n{'accuracy by language':<22}" + "".join(f"{e:>13}" for e in table))
    for language in ("en", "tr", "de", "fr"):
        cells = "".join(f"{r['language'][language]['accuracy']:>13.0%}" for r in table.values())
        print(f"{language:<22}{cells}")
    print(f"\n{'accuracy by category':<22}" + "".join(f"{e:>13}" for e in table))
    for category in next(iter(table.values()))["category"]:
        cells = "".join(f"{r['category'][category]:>13.0%}" for r in table.values())
        print(f"{category:<22}{cells}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "sensitivity.json"
    out.write_text(json.dumps(table, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--engine", required=True, choices=ENGINES)
    p.add_argument("--llm-model", default="qwen3:14b")
    sub.add_parser("score")
    args = parser.parse_args()
    {"predict": predict, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()

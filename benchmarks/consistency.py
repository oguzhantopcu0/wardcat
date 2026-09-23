"""Does one entity get one value across Turkish grammatical cases?

Turkish attaches case endings to names after an apostrophe ("Ahmet Yılmaz'ın"),
and spaCy includes the ending in the entity. An index built from redacted text
then sees a new value in every sentence. Each person and organisation below is
written six times — bare, and with the genitive, dative, ablative, instrumental
and accusative endings — and the script counts how many distinct values an engine
returned per entity. One is ideal.

    python benchmarks/consistency.py --engine wardcat
    python benchmarks/consistency.py --engine wardcat --ner-model tr_core_news_lg
    python benchmarks/consistency.py --engine presidio          # in a Presidio environment
"""

from __future__ import annotations

import argparse
import json

import compare

SENTENCES = {
    "bare": "Bugün {x} ile görüştük ve not aldık.",
    "genitive": "{x}{s} dosyası yarın kapanacak.",
    "dative": "Raporu {x}{s} e-posta ile gönderdik.",
    "ablative": "Onay {x}{s} geldi, süreç başladı.",
    "instrumental": "Toplantıyı {x}{s} birlikte planladık.",
    "accusative": "Müşteri temsilcisi dün {x}{s} aradı.",
}
CASES = ["genitive", "dative", "ablative", "instrumental", "accusative"]

ENTITIES = [
    ("PERSON", "Ahmet Yılmaz", ["'ın", "'a", "'dan", "'la", "'ı"]),
    ("PERSON", "Ayşe Demir", ["'in", "'e", "'den", "'le", "'i"]),
    ("PERSON", "Mehmet Öztürk", ["'ün", "'e", "'ten", "'le", "'ü"]),
    ("PERSON", "Zeynep Kaya", ["'nın", "'ya", "'dan", "'yla", "'yı"]),
    ("PERSON", "Can Arslan", ["'ın", "'a", "'dan", "'la", "'ı"]),
    ("PERSON", "Elif Şahin", ["'in", "'e", "'den", "'le", "'i"]),
    ("ORGANIZATION", "Garanti Bankası", ["'nın", "'na", "'ndan", "'yla", "'nı"]),
    ("ORGANIZATION", "Türk Telekom", ["'un", "'a", "'dan", "'la", "'u"]),
    ("ORGANIZATION", "Koç Holding", ["'in", "'e", "'den", "'le", "'i"]),
]


def samples():
    for gold_type, name, suffixes in ENTITIES:
        yield gold_type, name, SENTENCES["bare"].format(x=name)
        for case, suffix in zip(CASES, suffixes, strict=True):
            yield gold_type, name, SENTENCES[case].format(x=name, s=suffix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, choices=["presidio", "wardcat"])
    parser.add_argument("--ner-model", default=compare.NER_MODEL["tr"])
    args = parser.parse_args()

    compare.NER_MODEL["tr"] = args.ner_model
    run, info = compare.build_engine(args.engine, "tr", "")
    column = 1 if args.engine == "presidio" else 0

    per_entity: dict[str, dict] = {}
    for gold_type, name, text in samples():
        preds, _ = run(text)
        start = text.index(name)
        end = start + len(name)
        wanted = compare.SCORED[gold_type][column]
        hits = [text[s:e] for t, s, e in preds if t == wanted and s < end and start < e]
        record = per_entity.setdefault(name, {"detected": 0, "values": set()})
        record["detected"] += bool(hits)
        record["values"].update(hits)

    print(f"{args.engine} · {args.ner_model} · {json.dumps(info)}")
    for name, record in per_entity.items():
        values = sorted(record["values"])
        print(
            f"  {name:<16} found in {record['detected']}/6 cases, {len(values)} distinct: {values}"
        )
    distinct = [len(r["values"]) for r in per_entity.values()]
    single = sum(1 for d in distinct if d == 1)
    print(f"mean distinct values {sum(distinct) / len(distinct):.1f}, one value: {single}/9")


if __name__ == "__main__":
    main()

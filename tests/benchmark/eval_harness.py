"""
Precision / recall evaluation harness.

The false-positive suite (``test_false_positives.py``) only proves the detectors
stay quiet on clean text. This harness measures the other half — **recall** — by
running the guard over a labelled corpus and scoring detected spans against gold
annotations, reporting per-entity precision / recall / F1 plus a micro-average.

The corpus is deliberately restricted to **deterministic, regex-detectable**
entities across several languages, so the score is reproducible in CI with no
model downloads. All gold values are real and pass their checksums (card / IBAN /
TC), so a recall miss means a genuine detector regression — not bad fixtures.

Run as a script for a human-readable report::

    uv run python -m tests.benchmark.eval_harness
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from wardcat.detectors.regex_detector import RegexDetector

# ── Labelled corpus ───────────────────────────────────────────────────────────
# Each sample: the text, and the gold (entity_type, exact-substring) spans it
# contains. Clean samples (empty gold) exercise precision.
Sample = tuple[str, list[tuple[str, str]]]

CORPUS: list[Sample] = [
    # ── Turkish ──
    (
        "Müşteri TC 62601815964, IBAN TR330006100519786457841326 ve "
        "telefon +90 532 123 45 67 üzerinden ulaşıldı.",
        [
            ("TC_ID", "62601815964"),
            ("IBAN", "TR330006100519786457841326"),
            ("PHONE", "+90 532 123 45 67"),
        ],
    ),
    (
        "Kart numarası 4111111111111111 ve e-posta ali.veli@example.com.tr kaydedildi.",
        [
            ("CREDIT_CARD", "4111111111111111"),
            ("EMAIL", "ali.veli@example.com.tr"),
        ],
    ),
    (
        "Toplantı yarın saat 14:00'te yapılacak, herhangi bir kişisel veri yok.",
        [],  # clean
    ),
    # ── English ──
    (
        "Reach me at john.doe@company.com or on +44 20 7946 0958.",
        [
            ("EMAIL", "john.doe@company.com"),
            ("PHONE", "+44 20 7946 0958"),
        ],
    ),
    (
        "SSN 123-45-6789 and card 5555555555554444 on file.",
        [
            ("SSN", "123-45-6789"),
            ("CREDIT_CARD", "5555555555554444"),
        ],
    ),
    (
        "The quarterly report is attached for your review.",
        [],  # clean
    ),
    # ── German ──
    (
        "Meine IBAN ist DE89370400440532013000 und Amex 378282246310005.",
        [
            ("IBAN", "DE89370400440532013000"),
            ("CREDIT_CARD", "378282246310005"),
        ],
    ),
    # ── French ──
    (
        "Contactez-moi à marie.dupont@exemple.fr — IBAN FR1420041010050500013M02606.",
        [
            ("EMAIL", "marie.dupont@exemple.fr"),
            ("IBAN", "FR1420041010050500013M02606"),
        ],
    ),
    # ── Mixed / secrets ──
    (
        "Deploy key ghp_1234567890abcdefghijABCDEFGHIJ1234 leaked in logs.",
        [("CUSTOM_SECRET", "ghp_1234567890abcdefghijABCDEFGHIJ1234")],
    ),
    (
        "Server at 192.168.10.24 responded 200 OK in 12ms.",
        [("IP_ADDRESS", "192.168.10.24")],
    ),
    # ── Homoglyph (anti-evasion regression) ──
    (
        "Confidential contact: john.doe@cοmpany.com",  # Greek ο in domain
        [("EMAIL", "john.doe@cοmpany.com")],
    ),
]

# Every entity type referenced in the gold — the set the detector is asked for.
_GOLD_ENTITIES: set[str] = {etype for _, spans in CORPUS for etype, _ in spans}


@dataclass
class EntityScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class Report:
    per_entity: dict[str, EntityScore]

    @property
    def micro(self) -> EntityScore:
        agg = EntityScore()
        for s in self.per_entity.values():
            agg.tp += s.tp
            agg.fp += s.fp
            agg.fn += s.fn
        return agg

    def format_table(self) -> str:
        rows = [f"{'ENTITY':<16}{'P':>7}{'R':>7}{'F1':>7}{'TP':>5}{'FP':>5}{'FN':>5}"]
        rows.append("-" * 52)
        for etype in sorted(self.per_entity):
            s = self.per_entity[etype]
            rows.append(
                f"{etype:<16}{s.precision:>7.2f}{s.recall:>7.2f}{s.f1:>7.2f}"
                f"{s.tp:>5}{s.fp:>5}{s.fn:>5}"
            )
        m = self.micro
        rows.append("-" * 52)
        rows.append(
            f"{'MICRO-AVG':<16}{m.precision:>7.2f}{m.recall:>7.2f}{m.f1:>7.2f}"
            f"{m.tp:>5}{m.fp:>5}{m.fn:>5}"
        )
        return "\n".join(rows)


def evaluate(corpus: list[Sample] | None = None) -> Report:
    """Score the regex detector over ``corpus`` and return per-entity metrics.

    A prediction matches a gold span when both the entity type and the exact
    detected substring agree (multiset match, so repeats are counted).
    """
    corpus = corpus if corpus is not None else CORPUS
    detector = RegexDetector(_GOLD_ENTITIES)
    scores: dict[str, EntityScore] = {e: EntityScore() for e in _GOLD_ENTITIES}

    for text, gold_spans in corpus:
        gold = Counter(gold_spans)
        predicted = Counter((s.entity_type, s.text) for s in detector.detect(text))

        for key in gold | predicted:  # union of all (type, value) keys
            etype = key[0]
            g, p = gold[key], predicted[key]
            score = scores.setdefault(etype, EntityScore())
            score.tp += min(g, p)
            score.fp += max(p - g, 0)
            score.fn += max(g - p, 0)

    return Report(per_entity=scores)


def main() -> None:
    report = evaluate()
    print("wardcat regex detector — precision/recall\n")
    print(report.format_table())


if __name__ == "__main__":
    main()

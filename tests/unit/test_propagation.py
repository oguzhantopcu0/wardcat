"""Value propagation: once any layer detects a value, redact every occurrence.

Closes the gap where a model-based layer (NER/LLM) reports a repeated
value only once. Tested at the engine level with a fixed fake detector, plus
the Wardcat config wiring.
"""

from __future__ import annotations

from wardcat.core.engine import DetectionEngine
from wardcat.detectors.base import BaseDetector, DetectedSpan


class _Fixed(BaseDetector):
    def __init__(self, spans: list[DetectedSpan]) -> None:
        self._spans = spans

    def detect(self, text, candidates=None):
        return list(self._spans)


def _engine(spans, actions, **cfg) -> DetectionEngine:
    config = {
        "salt": "test",
        "entities": {k: {"enabled": True, "action": v} for k, v in actions.items()},
        **cfg,
    }
    return DetectionEngine(config, [_Fixed(spans)])


TEXT = "Ali went home. Ali came back. Alice waved."


class TestPropagation:
    def _person_span(self):
        i = TEXT.index("Ali")
        return DetectedSpan("PERSON", "Ali", i, i + 3, confidence=0.85)

    def test_off_by_default_only_detected_occurrence(self):
        res = _engine([self._person_span()], {"PERSON": "redact"}).scan(TEXT)
        persons = [v for v in res.violations if v.entity_type == "PERSON"]
        assert len(persons) == 1

    def test_on_redacts_every_whole_token_occurrence(self):
        res = _engine([self._person_span()], {"PERSON": "redact"}, propagate_matches=True).scan(
            TEXT
        )
        persons = [v for v in res.violations if v.entity_type == "PERSON"]
        assert len(persons) == 2  # both standalone "Ali"
        assert res.sanitized_text.count("[PERSON]") == 2

    def test_does_not_match_inside_larger_word(self):
        # "Alice" contains "Ali" but must NOT be redacted.
        res = _engine([self._person_span()], {"PERSON": "redact"}, propagate_matches=True).scan(
            TEXT
        )
        assert "Alice waved" in res.sanitized_text

    def test_min_length_skips_short_values(self):
        text = "Al and Al again."
        span = DetectedSpan("X", "Al", 0, 2, confidence=0.9)
        res = _engine([span], {"X": "redact"}, propagate_matches=True, propagate_min_length=3).scan(
            text
        )
        assert len(res.violations) == 1  # "Al" too short → not propagated

    def test_propagated_span_inherits_type_and_action(self):
        text = "secret42 here and secret42 there"
        i = text.index("secret42")
        span = DetectedSpan("CUSTOM_SECRET", "secret42", i, i + 8, confidence=0.85)
        res = _engine([span], {"CUSTOM_SECRET": "hash"}, propagate_matches=True).scan(text)
        secrets = [v for v in res.violations if v.entity_type == "CUSTOM_SECRET"]
        assert len(secrets) == 2
        assert all(v.action == "hash" for v in secrets)
        # Both occurrences hashed to the same token (deterministic value).
        assert "secret42" not in res.sanitized_text

    def test_deterministic_span_wins_over_propagated(self):
        # A checksum-regex value (conf 1.0) overlapping a propagated PERSON copy
        # keeps its own type. Here a high-conf EMAIL at one spot must not be
        # relabelled by a propagated lower-conf span of the same text.
        text = "code ABCD and code ABCD"
        i1 = text.index("ABCD")
        i2 = text.index("ABCD", i1 + 1)
        weak = DetectedSpan("PERSON", "ABCD", i1, i1 + 4, confidence=0.85)
        strong = DetectedSpan("CUSTOM_SECRET", "ABCD", i2, i2 + 4, confidence=1.0)
        res = _engine(
            [weak, strong],
            {"PERSON": "redact", "CUSTOM_SECRET": "hash"},
            propagate_matches=True,
        ).scan(text)
        # The strong span keeps its own type at its position; not overwritten.
        by_pos = {v.start: v.entity_type for v in res.violations}
        assert by_pos[i2] == "CUSTOM_SECRET"


class TestGuardWiring:
    def test_with_propagation_sets_config(self):
        from wardcat import Wardcat

        guard = Wardcat(use_ner=False).add_entity("EMAIL", "redact").with_propagation(min_length=4)
        assert guard._config["propagate_matches"] is True
        assert guard._config["propagate_min_length"] == 4

    def test_with_propagation_disabled(self):
        from wardcat import Wardcat

        guard = Wardcat(use_ner=False).with_propagation(enabled=False)
        assert guard._config["propagate_matches"] is False

    def test_end_to_end_regex_entity_all_occurrences(self):
        # EMAIL is regex (finds all anyway) — this checks propagation does not
        # break or double-count a value regex already covers exhaustively.
        from wardcat import Wardcat

        guard = Wardcat(salt="s", use_ner=False).add_entity("EMAIL", "redact").with_propagation()
        res = guard.scan("Mail a@b.com now, or a@b.com later.")
        emails = [v for v in res.violations if v.entity_type == "EMAIL"]
        assert len(emails) == 2
        assert res.sanitized_text.count("[EMAIL]") == 2

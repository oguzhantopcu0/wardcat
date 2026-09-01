"""Reversible masking: the ``tokenize`` action and ``ScanResult.restore()``."""

from __future__ import annotations

import pytest

from wardcat import Action, Entity, RestoredText, Violation, Wardcat
from wardcat.core.actions import TokenAllocator
from wardcat.core.restore import restore_text

EMAIL_A = "ali@example.com"
EMAIL_B = "veli@example.com"
CARD = "4532 0151 1283 0366"


@pytest.fixture
def guard() -> Wardcat:
    """Regex-only guard (no models) with the reversible action on two types."""
    return (
        Wardcat(salt="test-salt")
        .add_entity(Entity.EMAIL, Action.TOKENIZE)
        .add_entity(Entity.CREDIT_CARD, Action.TOKENIZE)
    )


def _violation(entity_type: str, original: str, replacement: str | None, **kw: object) -> Violation:
    return Violation(
        entity_type=entity_type,
        original=original,
        start=kw.get("start", 0),  # type: ignore[arg-type]
        end=kw.get("end", 0),  # type: ignore[arg-type]
        action=kw.get("action", "tokenize"),  # type: ignore[arg-type]
        replacement=replacement,
        confidence=kw.get("confidence", 1.0),  # type: ignore[arg-type]
    )


# ── The tokenize action ───────────────────────────────────────────────────────


class TestTokenizeAction:
    def test_registered_as_a_built_in(self) -> None:
        from wardcat import registered_actions

        assert "tokenize" in registered_actions()
        assert Action.TOKENIZE.value == "tokenize"

    def test_replaces_values_with_numbered_placeholders(self, guard: Wardcat) -> None:
        result = guard.scan(f"mail {EMAIL_A}, card {CARD}")

        assert result.sanitized_text == "mail [EMAIL_1], card [CREDIT_CARD_1]"
        assert EMAIL_A not in result.sanitized_text
        assert CARD not in result.sanitized_text

    def test_numbers_per_entity_type_in_order_of_appearance(self, guard: Wardcat) -> None:
        result = guard.scan(f"{EMAIL_A} then {EMAIL_B} then {CARD}")

        assert result.sanitized_text == "[EMAIL_1] then [EMAIL_2] then [CREDIT_CARD_1]"

    def test_same_value_keeps_the_same_token(self, guard: Wardcat) -> None:
        """A repeated value stays one referent — an LLM can still co-refer it."""
        result = guard.scan(f"{EMAIL_A} wrote to {EMAIL_B}; reply to {EMAIL_A}")

        assert result.sanitized_text == "[EMAIL_1] wrote to [EMAIL_2]; reply to [EMAIL_1]"

    def test_numbering_restarts_for_each_scan(self, guard: Wardcat) -> None:
        """State must not leak between scans through the shared Anonymizer."""
        first = guard.scan(f"mail {EMAIL_A}")
        second = guard.scan(f"mail {EMAIL_B}")

        assert first.sanitized_text == second.sanitized_text == "mail [EMAIL_1]"
        assert first.token_map == {"[EMAIL_1]": EMAIL_A}
        assert second.token_map == {"[EMAIL_1]": EMAIL_B}

    def test_concurrent_scans_do_not_share_a_counter(self, guard: Wardcat) -> None:
        texts = [f"mail {EMAIL_A}", f"mail {EMAIL_B}", f"card {CARD}"]

        results = guard.scan_batch(texts, max_workers=3)

        assert [r.sanitized_text for r in results] == [
            "mail [EMAIL_1]",
            "mail [EMAIL_1]",
            "card [CREDIT_CARD_1]",
        ]

    def test_allocator_is_deterministic_and_stateful(self) -> None:
        allocator = TokenAllocator()

        assert allocator.token_for("PERSON", "Ali") == "[PERSON_1]"
        assert allocator.token_for("PERSON", "Veli") == "[PERSON_2]"
        assert allocator.token_for("PERSON", "Ali") == "[PERSON_1]"
        assert allocator.token_for("EMAIL", "Ali") == "[EMAIL_1]"

    def test_action_context_equality_ignores_the_allocator(self) -> None:
        from wardcat import ActionContext

        assert ActionContext(salt="s") == ActionContext(salt="s")
        assert "tokens" not in repr(ActionContext(salt="s"))


class TestEveryDetectionLayer:
    """The action runs after detection, so the layer that found a span is irrelevant."""

    def test_an_llm_detected_span_is_tokenized_and_restorable(self) -> None:
        from unittest.mock import MagicMock

        from wardcat.llm.backends.base import Backend, BaseLLMBackend

        backend = MagicMock(spec=BaseLLMBackend)
        backend.complete_messages.return_value = (
            '[{"type": "SPECIAL_CATEGORY", "text": "Type 1 diabetes"}]'
        )
        guard = (
            Wardcat(salt="s")
            .with_llm(backend=Backend.OLLAMA, model="stub")
            .add_entity(Entity.SPECIAL_CATEGORY, Action.TOKENIZE, layers=["llm"])
            .add_entity(Entity.EMAIL, Action.TOKENIZE)
        )
        guard._engine.detectors[-1].backend = backend

        result = guard.scan(f"Patient reports Type 1 diabetes; contact {EMAIL_A}")

        assert result.sanitized_text == ("Patient reports [SPECIAL_CATEGORY_1]; contact [EMAIL_1]")
        assert result.token_map == {
            "[SPECIAL_CATEGORY_1]": "Type 1 diabetes",
            "[EMAIL_1]": EMAIL_A,
        }

        restored = result.restore("Regarding [SPECIAL_CATEGORY_1], I emailed [EMAIL_1].")

        assert restored.text == f"Regarding Type 1 diabetes, I emailed {EMAIL_A}."
        # The model-based span keeps its lower confidence in the source list.
        assert restored.substitutions[0].confidence == 0.85
        assert restored.substitutions[1].confidence == 0.97


# ── token_map ─────────────────────────────────────────────────────────────────


class TestTokenMap:
    def test_maps_placeholders_to_originals(self, guard: Wardcat) -> None:
        result = guard.scan(f"mail {EMAIL_A}, card {CARD}")

        assert result.token_map == {"[EMAIL_1]": EMAIL_A, "[CREDIT_CARD_1]": CARD}

    def test_excludes_report_only_violations(self) -> None:
        guard = Wardcat(salt="s").add_entity(Entity.EMAIL, Action.WARN)

        assert guard.scan(f"mail {EMAIL_A}").token_map == {}

    def test_empty_for_a_clean_scan(self, guard: Wardcat) -> None:
        assert guard.scan("nothing to see here").token_map == {}


# ── restore() ─────────────────────────────────────────────────────────────────


class TestRestore:
    def test_round_trips_the_sanitized_text_by_default(self, guard: Wardcat) -> None:
        text = f"mail {EMAIL_A}, card {CARD}"

        assert guard.scan(text).restore().text == text

    def test_restores_placeholders_in_an_llm_answer(self, guard: Wardcat) -> None:
        result = guard.scan(f"mail {EMAIL_A}, card {CARD}")

        restored = result.restore("I have emailed [EMAIL_1] about card [CREDIT_CARD_1].")

        assert restored.text == f"I have emailed {EMAIL_A} about card {CARD}."
        assert restored.is_complete

    def test_reports_substitutions_in_order_of_appearance(self, guard: Wardcat) -> None:
        result = guard.scan(f"mail {EMAIL_A}, card {CARD}")

        restored = result.restore("Card [CREDIT_CARD_1] belongs to [EMAIL_1].")

        assert [(s.index, s.placeholder) for s in restored.substitutions] == [
            (1, "[CREDIT_CARD_1]"),
            (2, "[EMAIL_1]"),
        ]
        assert restored.substitutions[0].entity_type == "CREDIT_CARD"
        assert restored.substitutions[0].action == "tokenize"
        assert restored.substitutions[1].original == EMAIL_A

    def test_counts_occurrences(self, guard: Wardcat) -> None:
        result = guard.scan(f"mail {EMAIL_A}")

        restored = result.restore("[EMAIL_1] and [EMAIL_1] and [EMAIL_1]")

        assert restored.text == f"{EMAIL_A} and {EMAIL_A} and {EMAIL_A}"
        assert restored.substitutions[0].occurrences == 3

    def test_placeholder_absent_from_the_text_is_reported_not_present(self, guard: Wardcat) -> None:
        result = guard.scan(f"mail {EMAIL_A}, card {CARD}")

        restored = result.restore("I could not help with that.")

        assert restored.text == "I could not help with that."
        assert restored.substitutions == []
        assert {u.reason for u in restored.unrestored} == {"not-present"}
        assert restored.is_complete  # nothing to put back is not a failure

    def test_report_only_violations_are_reported_not_replaced(self) -> None:
        guard = Wardcat(salt="s").add_entity(Entity.EMAIL, Action.WARN)

        restored = guard.scan(f"mail {EMAIL_A}").restore()

        assert restored.text == f"mail {EMAIL_A}"
        assert [u.reason for u in restored.unrestored] == ["not-replaced"]
        assert restored.unrestored[0].placeholder is None

    def test_hashed_values_are_restorable(self) -> None:
        guard = Wardcat(salt="s").add_entity(Entity.EMAIL, Action.HASH)
        result = guard.scan(f"mail {EMAIL_A}")

        restored = result.restore()

        assert restored.text == f"mail {EMAIL_A}"
        assert restored.substitutions[0].action == "hash"

    def test_a_placeholder_standing_for_two_values_is_left_alone(self) -> None:
        """``redact`` collapses values onto one label — restoring would be a guess."""
        guard = Wardcat(salt="s").add_entity(Entity.EMAIL, Action.REDACT)
        result = guard.scan(f"{EMAIL_A} and {EMAIL_B}")
        assert result.sanitized_text == "[EMAIL] and [EMAIL]"

        restored = result.restore()

        assert restored.text == "[EMAIL] and [EMAIL]"
        assert [u.reason for u in restored.unrestored] == ["ambiguous"]
        assert not restored.is_complete

    def test_an_unambiguous_redaction_is_restored(self) -> None:
        guard = Wardcat(salt="s").add_entity(Entity.EMAIL, Action.REDACT)

        restored = guard.scan(f"mail {EMAIL_A}").restore()

        assert restored.text == f"mail {EMAIL_A}"

    def test_reapply_derives_a_reversible_view_of_any_scan(self) -> None:
        guard = Wardcat(salt="s").add_entity(Entity.EMAIL, Action.MASK)
        result = guard.scan(f"{EMAIL_A} and {EMAIL_B}")

        reversible = result.reapply(Action.TOKENIZE)

        assert reversible.sanitized_text == "[EMAIL_1] and [EMAIL_2]"
        assert reversible.restore().text == f"{EMAIL_A} and {EMAIL_B}"

    def test_a_restored_value_is_never_re_matched(self) -> None:
        """One left-to-right pass: a value that looks like a placeholder is inert."""
        violations = [
            _violation("PERSON", "[EMAIL_1]", "[PERSON_1]"),
            _violation("EMAIL", EMAIL_A, "[EMAIL_1]"),
        ]

        restored = restore_text("[PERSON_1] wrote it", violations)

        assert restored.text == "[EMAIL_1] wrote it"

    def test_longer_placeholders_win_over_their_prefixes(self) -> None:
        violations = [
            _violation("PERSON", "Ali", "ab**ef", action="mask"),
            _violation("PERSON", "Ali Veli", "ab**efgh", action="mask"),
        ]

        restored = restore_text("ab**efgh met ab**ef", violations)

        assert restored.text == "Ali Veli met Ali"

    def test_an_empty_replacement_is_never_matched(self) -> None:
        """A custom action may delete a value; nothing can locate it afterwards."""
        restored = restore_text("some answer", [_violation("EMAIL", EMAIL_A, "")])

        assert restored.text == "some answer"
        assert [u.reason for u in restored.unrestored] == ["not-present"]

    def test_clean_scan_leaves_the_text_untouched(self, guard: Wardcat) -> None:
        restored = guard.scan("nothing here").restore("an answer")

        assert restored.text == "an answer"
        assert restored.substitutions == []
        assert restored.unrestored == []


class TestEchoRoundTrip:
    """The full LLM round trip with a stub model that echoes the masked text back.

    No network, no model weights: the "model" returns exactly what it was given,
    which is the strictest possible round trip — every placeholder comes back, so
    restoring must reproduce the input byte for byte.
    """

    PROMPT = (
        "Customer Jonathan Blake wrote from jonathan.blake@example.com "
        "about card 4532 0151 1283 0366; reply to jonathan.blake@example.com. "
        "Refund to TR33 0006 1005 1978 6457 8413 26."
    )
    SECRETS = [
        "jonathan.blake@example.com",
        "4532 0151 1283 0366",
        "TR33 0006 1005 1978 6457 8413 26",
    ]

    @staticmethod
    def echo_llm(prompt: str) -> str:
        """Stands in for the model: returns the masked text unchanged."""
        return prompt

    @pytest.fixture
    def result(self):
        guard = Wardcat(salt="round-trip").add_entities(
            [Entity.EMAIL, Entity.CREDIT_CARD, Entity.IBAN], action=Action.TOKENIZE
        )
        return guard.scan(self.PROMPT)

    def test_no_raw_value_reaches_the_model(self, result) -> None:
        sent = result.sanitized_text

        for secret in self.SECRETS:
            assert secret not in sent, secret
        assert sent == (
            "Customer Jonathan Blake wrote from [EMAIL_1] "
            "about card [CREDIT_CARD_1]; reply to [EMAIL_1]. "
            "Refund to [IBAN_1]."
        )

    def test_echoed_answer_restores_to_the_original_byte_for_byte(self, result) -> None:
        answer = self.echo_llm(result.sanitized_text)

        restored = result.restore(answer)

        assert restored.text == self.PROMPT
        assert restored.is_complete
        assert restored.unrestored == []

    def test_every_masked_value_is_cited_in_order(self, result) -> None:
        restored = result.restore(self.echo_llm(result.sanitized_text))

        assert [(s.index, s.placeholder, s.occurrences) for s in restored.substitutions] == [
            (1, "[EMAIL_1]", 2),  # the repeated address is one token, cited once
            (2, "[CREDIT_CARD_1]", 1),
            (3, "[IBAN_1]", 1),
        ]
        block = restored.sources_block()
        for secret in self.SECRETS:
            assert secret in block, secret

    def test_the_round_trip_survives_a_second_pass(self, result) -> None:
        """Restoring an already-restored text is a no-op, not a double substitution."""
        once = result.restore(self.echo_llm(result.sanitized_text))

        twice = result.restore(once.text)

        assert twice.text == self.PROMPT
        assert twice.substitutions == []  # nothing left to put back


# ── The source list ───────────────────────────────────────────────────────────


class TestSourcesBlock:
    def test_lists_each_substitution_in_order(self, guard: Wardcat) -> None:
        result = guard.scan(f"mail {EMAIL_A}, card {CARD}")

        block = result.restore("[EMAIL_1] paid with [CREDIT_CARD_1].").sources_block()

        lines = block.splitlines()
        assert lines[0] == "--- Sources ---"
        assert lines[1] == f"[1] [EMAIL_1] → {EMAIL_A} (EMAIL · tokenize · confidence 0.97)"
        assert (
            lines[2] == f"[2] [CREDIT_CARD_1] → {CARD} (CREDIT_CARD · tokenize · confidence 1.00)"
        )

    def test_marks_repeated_placeholders_with_a_count(self, guard: Wardcat) -> None:
        result = guard.scan(f"mail {EMAIL_A}")

        block = result.restore("[EMAIL_1] and [EMAIL_1]").sources_block()

        assert "x2" in block

    def test_notes_the_values_that_were_not_mentioned(self, guard: Wardcat) -> None:
        result = guard.scan(f"mail {EMAIL_A}, card {CARD}")

        block = result.restore("Emailed [EMAIL_1].").sources_block()

        assert "1 masked value(s) not mentioned here: CREDIT_CARD." in block

    def test_notes_can_be_turned_off(self, guard: Wardcat) -> None:
        result = guard.scan(f"mail {EMAIL_A}, card {CARD}")

        block = result.restore("Emailed [EMAIL_1].").sources_block(notes=False)

        assert "not mentioned" not in block

    def test_notes_ambiguous_placeholders(self) -> None:
        guard = Wardcat(salt="s").add_entity(Entity.EMAIL, Action.REDACT)

        block = guard.scan(f"{EMAIL_A} and {EMAIL_B} and {CARD}").restore().sources_block()

        assert "left in place: [EMAIL]" in block
        assert "tokenize" in block  # points at the fix

    def test_title_is_configurable(self, guard: Wardcat) -> None:
        result = guard.scan(f"mail {EMAIL_A}")

        assert result.restore().sources_block(title="Kaynaklar").startswith("--- Kaynaklar ---")

    def test_empty_when_nothing_was_substituted(self, guard: Wardcat) -> None:
        assert guard.scan("nothing here").restore().sources_block() == ""

    def test_str_appends_the_block_below_the_text(self, guard: Wardcat) -> None:
        result = guard.scan(f"mail {EMAIL_A}")

        rendered = str(result.restore("Emailed [EMAIL_1]."))

        assert rendered.startswith(f"Emailed {EMAIL_A}.\n\n--- Sources ---")
        assert rendered == result.restore("Emailed [EMAIL_1].").with_sources()

    def test_str_is_just_the_text_when_there_are_no_sources(self, guard: Wardcat) -> None:
        assert str(guard.scan("nothing here").restore("an answer")) == "an answer"

    def test_repr_hides_the_values(self, guard: Wardcat) -> None:
        result = guard.scan(f"mail {EMAIL_A}")

        assert repr(result.restore()) == (
            "RestoredText(substitutions=1, unrestored=0, is_complete=True)"
        )
        assert EMAIL_A not in repr(result.restore())


def test_restored_text_is_exported() -> None:
    import wardcat

    assert wardcat.RestoredText is RestoredText
    assert {"RestoredText", "Substitution", "UnrestoredValue", "TokenAllocator"} <= set(
        wardcat.__all__
    )

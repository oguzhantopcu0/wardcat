"""Every violation names the layer that found it."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from wardcat import Action, Entity, Layer, Wardcat
from wardcat.detectors.base import BaseDetector, DetectedSpan
from wardcat.llm.backends.base import BaseLLMBackend


class TestBuiltInLayers:
    def test_regex_and_denylist_and_propagation(self) -> None:
        guard = (
            Wardcat(salt="s")
            .add_entity(Entity.EMAIL, Action.REDACT)
            .add_denylist([{"value": "ProjectX", "entity_type": "CUSTOM_SECRET"}])
            .with_propagation()
        )
        # The e-mail appears twice: regex finds both, so nothing is left to propagate
        # for it. ProjectX is a denylist hit; "Acme Corp" is propagated from a
        # denylist value found once.
        guard.add_denylist([{"value": "Acme Corp", "entity_type": "ORG"}])
        result = guard.scan("ali@example.com about ProjectX at Acme Corp; Acme Corp again")

        by_text = {(v.original, v.start): v.source for v in result.violations}
        assert by_text[("ali@example.com", 0)] == "regex"
        assert by_text[("ProjectX", 22)] == "denylist"
        sources = sorted(v.source for v in result.violations if v.original == "Acme Corp")
        assert sources == ["denylist", "denylist"]

    def test_propagated_copies_are_marked(self) -> None:
        class OnceDetector(BaseDetector):
            layer = "ner"

            def detect(self, text, candidates=None):
                i = text.index("Ada Lovelace")
                return [DetectedSpan("PERSON", "Ada Lovelace", i, i + 12, 0.85)]

        guard = Wardcat(salt="s").add_entity(Entity.PERSON, Action.REDACT).with_propagation()
        guard._engine.detectors.append(OnceDetector())
        result = guard.scan("Ada Lovelace wrote; Ada Lovelace signed.")

        assert [v.source for v in result.violations] == ["ner", "propagation"]

    def test_llm_spans_and_reapply_keep_the_source(self) -> None:
        backend = MagicMock(spec=BaseLLMBackend)
        backend.complete_messages.return_value = json.dumps(
            {"entities": [{"type": "PERSON", "text": "Ada Lovelace"}]}
        )
        guard = (
            Wardcat(salt="s")
            .with_llm(model="x")
            .add_entity(Entity.PERSON, Action.REDACT, layers=[Layer.LLM])
        )
        guard._engine.detectors[-1].backend = backend
        result = guard.scan("Ada Lovelace wrote.")

        assert [v.source for v in result.violations] == ["llm"]
        assert [v.source for v in result.reapply(Action.HASH).violations] == ["llm"]
        assert result.redacted()["violations"][0]["source"] == "llm"

    def test_a_third_party_detector_that_does_not_name_itself_is_custom(self) -> None:
        class Quiet(BaseDetector):
            def detect(self, text, candidates=None):
                return [DetectedSpan("EMAIL", "x@y.io", 0, 6, 0.97)]

        guard = Wardcat(salt="s").add_entity(Entity.EMAIL, Action.REDACT)
        guard._engine.detectors[:] = [Quiet()]

        assert guard.scan("x@y.io").violations[0].source == "custom"
        assert asyncio.run(guard.scan_async("x@y.io")).violations[0].source == "custom"

    def test_no_violation_is_left_without_a_source(self) -> None:
        guard = Wardcat(salt="s").add_entities(
            [Entity.EMAIL, Entity.CREDIT_CARD, Entity.PHONE], action=Action.REDACT
        )
        result = guard.scan("a@b.io, 4111 1111 1111 1111, Phone: 905-674-3793")

        assert result.violations
        assert all(v.source for v in result.violations)


class TestLayerEnum:
    def test_layers_accept_the_enum(self) -> None:
        guard = Wardcat(salt="s").add_entity(Entity.EMAIL, Action.REDACT, layers=[Layer.REGEX])
        assert guard.scan("x@y.io").sanitized_text == "[EMAIL]"

    def test_supported_entities_accepts_the_enum(self) -> None:
        assert Wardcat.supported_entities(Layer.NER) == Wardcat.supported_entities("ner")

    def test_an_unknown_layer_is_still_refused(self) -> None:
        from wardcat import ConfigError

        with pytest.raises(ConfigError):
            Wardcat(salt="s").add_entity(Entity.EMAIL, Action.REDACT, layers=["magic"])

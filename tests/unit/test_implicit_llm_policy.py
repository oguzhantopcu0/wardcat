"""``with_llm()`` brings its own entity policy — say so, once."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from wardcat import Action, Backend, Entity, Wardcat
from wardcat.llm.backends.base import BaseLLMBackend

_MARKER = "own default entity policy"


def _llm_guard(**kwargs: object) -> Wardcat:
    """A guard with the LLM layer on."""
    return Wardcat(salt="s", **kwargs).with_llm(  # type: ignore[arg-type]
        backend=Backend.OLLAMA, model="stub"
    )


def _stub(guard: Wardcat) -> Wardcat:
    """Swap in a backend that finds nothing — call last, a rebuild would drop it."""
    backend = MagicMock(spec=BaseLLMBackend)
    backend.complete_messages.return_value = "[]"
    guard._engine.detectors[-1].backend = backend
    return guard


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if _MARKER in r.getMessage()]


class TestImplicitLLMPolicyWarning:
    def test_warns_when_the_llm_layer_enables_entities_the_caller_did_not(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        guard = _stub(_llm_guard().add_entity(Entity.EMAIL, Action.HASH))

        with caplog.at_level(logging.WARNING, logger="wardcat.guard"):
            guard.scan("ali@example.com")

        assert len(_warnings(caplog)) == 1
        message = _warnings(caplog)[0]
        assert "PERSON (hash)" in message  # an entity nobody asked for
        assert "EMAIL" not in message  # the one that was configured
        assert "remove_entity" in message  # and how to switch it off

    def test_warns_only_once_per_guard(self, caplog: pytest.LogCaptureFixture) -> None:
        guard = _stub(_llm_guard().add_entity(Entity.EMAIL, Action.HASH))

        with caplog.at_level(logging.WARNING, logger="wardcat.guard"):
            guard.scan("ali@example.com")
            guard.scan("veli@example.com")
            guard.scan("ayse@example.com")

        assert len(_warnings(caplog)) == 1

    def test_ner_enables_nothing_so_it_never_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        guard = Wardcat(salt="s").with_ner(language="en").add_entity(Entity.EMAIL, Action.HASH)

        with caplog.at_level(logging.WARNING, logger="wardcat.guard"):
            guard.scan("ali@example.com")

        assert _warnings(caplog) == []

    def test_silent_once_every_llm_entity_is_the_caller_s_own(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        guard = _llm_guard()
        guard.remove_entity(Entity.ALL)
        guard.add_entity(Entity.SPECIAL_CATEGORY, Action.HASH, layers=["llm"])
        _stub(guard)

        with caplog.at_level(logging.WARNING, logger="wardcat.guard"):
            guard.scan("nothing here")

        assert _warnings(caplog) == []

    def test_silent_for_a_caller_who_supplied_a_policy_file(
        self, caplog: pytest.LogCaptureFixture, tmp_path
    ) -> None:
        """Everything in a policy file is a deliberate choice — nothing to point out."""
        policy = tmp_path / "policy.yaml"
        policy.write_text(
            "entities:\n  EMAIL: { enabled: true, action: hash }\n",
            encoding="utf-8",
        )
        guard = _stub(_llm_guard(config_path=str(policy)))

        with caplog.at_level(logging.WARNING, logger="wardcat.guard"):
            guard.scan("ali@example.com")

        assert _warnings(caplog) == []

    def test_taking_control_of_an_entity_removes_it_from_the_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        guard = _stub(_llm_guard().add_entity(Entity.PERSON, Action.REDACT, layers=["llm"]))

        with caplog.at_level(logging.WARNING, logger="wardcat.guard"):
            guard.scan("some text")

        assert "PERSON" not in _warnings(caplog)[0]

    def test_change_entity_action_also_counts_as_taking_control(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        guard = _llm_guard()
        guard.change_entity_action(Entity.PERSON, Action.REDACT)
        _stub(guard)

        with caplog.at_level(logging.WARNING, logger="wardcat.guard"):
            guard.scan("some text")

        assert "PERSON" not in _warnings(caplog)[0]

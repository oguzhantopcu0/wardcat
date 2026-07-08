"""Semantic sensitivity check — Wardcat.is_sensitive() (LLM-only, boolean)."""

from __future__ import annotations

import asyncio

import pytest

from wardcat import ConfigError, Wardcat
from wardcat.llm.backends.base import BaseLLMBackend
from wardcat.llm.prompt import parse_sensitivity


class _StubBackend(BaseLLMBackend):
    """Returns a canned reply and records how it was called."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.messages: list | None = None

    def complete(self, prompt, *, timeout=60):
        return self.reply

    def complete_messages(self, messages, *, timeout=60):
        self.messages = messages
        return self.reply

    async def complete_messages_async(self, messages, *, timeout=60):
        self.messages = messages
        return self.reply

    def list_models(self):
        return []

    def pull_model(self, model, *, on_progress=None):
        pass


class _DeadBackend(BaseLLMBackend):
    def complete(self, prompt, *, timeout=60):
        raise ConnectionError("backend down")

    def complete_messages(self, messages, *, timeout=60):
        raise ConnectionError("backend down")

    def list_models(self):
        return []

    def pull_model(self, model, *, on_progress=None):
        pass


def _guard_with(reply: str) -> Wardcat:
    guard = Wardcat(salt="s").with_llm(model="stub")
    guard._llm_detector.backend = _StubBackend(reply)  # type: ignore[union-attr]
    return guard


class TestIsSensitive:
    def test_true_reply_flags_sensitive(self):
        assert _guard_with("true").is_sensitive("db_pass=S3cr3t!42") is True

    def test_false_reply_flags_clean(self):
        assert _guard_with("false").is_sensitive("The weather is nice today.") is False

    def test_reply_with_extra_words_still_parses(self):
        assert _guard_with("false, no personal data here").is_sensitive("hi") is False

    def test_empty_text_short_circuits_to_false(self):
        guard = _guard_with("true")  # backend would say true, but text is blank
        assert guard.is_sensitive("   \n ") is False
        assert guard._llm_detector.backend.messages is None  # LLM not even called

    def test_sends_a_two_message_chat(self):
        guard = _guard_with("true")
        guard.is_sensitive("Ali Veli, TC 10987654321")
        msgs = guard._llm_detector.backend.messages
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert "sensitive" in msgs[0]["content"].lower()

    def test_requires_llm_layer(self):
        guard = Wardcat(salt="s")  # no with_llm
        with pytest.raises(ConfigError, match="with_llm"):
            guard.is_sensitive("anything")

    def test_fail_closed_propagates_backend_error(self):
        guard = Wardcat(salt="s").with_llm(model="stub")
        guard._llm_detector.backend = _DeadBackend()  # type: ignore[union-attr]
        with pytest.raises(ConnectionError):
            guard.is_sensitive("secret text")

    def test_async_variant(self):
        guard = _guard_with("true")
        assert asyncio.run(guard.is_sensitive_async("secret: api_key=abc")) is True


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("true", True),
        ("false", False),
        ("TRUE", True),
        ("False.", False),
        ("yes", True),
        ("no", False),
        ("evet", True),
        ("hayır", False),
        ("none found", False),
        ("true — contains an email", True),
        ("", True),  # no signal → cautious default
        ("maybe, unclear", True),  # ambiguous → cautious default
    ],
)
def test_parse_sensitivity(reply, expected):
    assert parse_sensitivity(reply) is expected

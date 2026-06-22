"""Tests for the pluggable LLM backend registry (Open/Closed extension point)."""

import pytest

from ai_guard import AIGuard, BaseLLMBackend, register_backend, registered_backends
from ai_guard.exceptions import ConfigError
from ai_guard.llm.backends.registry import create_backend


class _EchoBackend(BaseLLMBackend):
    """A minimal third-party backend used to prove the registry is extensible."""

    def __init__(self, model: str = "echo") -> None:
        self.model = model

    def complete(self, prompt, *, timeout=60):
        return "[]"

    def complete_messages(self, messages, *, timeout=60):
        return "[]"

    def list_models(self):
        return [self.model]

    def pull_model(self, model, *, on_progress=None):
        pass


def test_builtin_backends_registered():
    assert {"ollama", "openai_compatible", "transformers"} <= registered_backends()


def test_register_and_create_custom_backend():
    register_backend("echo_test", lambda cfg: _EchoBackend(cfg.get("model", "echo")))
    assert "echo_test" in registered_backends()
    backend = create_backend({"backend": "echo_test", "model": "m1"})
    assert isinstance(backend, _EchoBackend)
    assert backend.model == "m1"


def test_custom_backend_works_end_to_end():
    register_backend("echo_e2e", lambda cfg: _EchoBackend())
    guard = (
        AIGuard(salt="s", use_ner=False)
        .with_llm(backend="echo_e2e")
        .add_entity("EMAIL", "warn")
    )
    assert guard._config["llm_detector"]["backend"] == "echo_e2e"
    # Regex still works alongside the custom LLM backend (no crash on scan).
    assert guard.scan("mail a@b.com").sanitized_text == "mail a@b.com"


def test_unknown_backend_raises_with_registered_list():
    with pytest.raises(ConfigError, match="Registered backends"):
        AIGuard(salt="s").with_llm(backend="does_not_exist")


def test_create_backend_defaults_to_ollama():
    # No backend key → ollama factory; constructed without connecting.
    backend = create_backend({"base_url": "http://localhost:11434"})
    assert type(backend).__name__ == "OllamaBackend"

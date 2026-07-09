"""Tests for LLM backend selection — built-in backends only (not user-extensible)."""

import pytest

import wardcat
from wardcat import Wardcat
from wardcat.exceptions import ConfigError
from wardcat.llm.backends.registry import create_backend, supported_backends


def test_builtin_backends_supported():
    assert supported_backends() == {"ollama", "openai_compatible", "vllm", "transformers"}


def test_create_selects_named_backend():
    backend = create_backend({"backend": "openai_compatible"})
    assert type(backend).__name__ == "OpenAICompatBackend"


def test_create_backend_defaults_to_ollama():
    # No backend key → ollama factory; constructed without connecting.
    backend = create_backend({"base_url": "http://localhost:11434"})
    assert type(backend).__name__ == "OllamaBackend"


def test_unknown_backend_raises_with_supported_list():
    with pytest.raises(ConfigError, match="Supported backends"):
        Wardcat(salt="s").with_llm(backend="does_not_exist")


def test_custom_backend_registration_is_removed():
    # Backends are not user-extensible; the registration helpers are gone.
    assert not hasattr(wardcat, "register_backend")
    assert not hasattr(wardcat, "registered_backends")
    assert not hasattr(wardcat, "BaseLLMBackend")

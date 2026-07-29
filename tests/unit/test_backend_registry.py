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


def _built_backend(guard):
    """The LLM backend a configured guard would use."""
    return create_backend(guard._config["llm_detector"])


@pytest.mark.parametrize(
    ("backend", "expected_base_url"),
    [
        ("ollama", "http://localhost:11434"),
        ("openai_compatible", "http://localhost:11434"),
        ("vllm", "http://localhost:8000/v1"),
    ],
)
def test_with_llm_uses_backend_default_base_url(backend, expected_base_url):
    # Regression: with_llm() used to hardcode base_url=http://localhost:11434,
    # so selecting vllm without a base_url silently hit Ollama's port instead
    # of vLLM's. Leaving base_url unset must defer to each backend's default.
    guard = Wardcat(salt="s").with_llm(backend=backend, model="m", allow_http=True)
    assert _built_backend(guard).base_url == expected_base_url


def test_with_llm_explicit_base_url_is_respected():
    guard = Wardcat(salt="s").with_llm(
        backend="vllm", model="m", base_url="http://gpu-host:9000/v1", allow_http=True
    )
    assert _built_backend(guard).base_url == "http://gpu-host:9000/v1"


def test_with_llm_reconfigure_resets_base_url_to_new_backend_default():
    # vllm (no base_url) → ollama (no base_url): must reset to Ollama's default,
    # not keep vLLM's 8000/v1 from the first call.
    guard = Wardcat(salt="s").with_llm(backend="vllm", model="m", allow_http=True)
    guard.with_llm(backend="ollama", model="llama3.2", allow_http=True)
    assert _built_backend(guard).base_url == "http://localhost:11434"


def test_unknown_backend_raises_with_supported_list():
    with pytest.raises(ConfigError, match="Supported backends"):
        Wardcat(salt="s").with_llm(backend="does_not_exist")


def test_custom_backend_registration_is_removed():
    # Backends are not user-extensible; the registration helpers are gone.
    assert not hasattr(wardcat, "register_backend")
    assert not hasattr(wardcat, "registered_backends")
    assert not hasattr(wardcat, "BaseLLMBackend")

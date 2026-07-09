"""LLM backend construction — one factory per built-in backend.

wardcat ships four backends — Ollama, OpenAI-compatible, vLLM and Transformers —
selected via the :class:`~wardcat.Backend` enum (or the ``backend`` config value).
Backends are **not** user-extensible on purpose: a third-party backend would sit
outside wardcat's safety checks (e.g. the plaintext-HTTP-to-remote guard) and its
PII-handling guarantees, which is exactly where sensitive data would leak. Point
one of the built-ins at your endpoint instead (``openai_compatible`` covers most
OpenAI-style gateways).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from wardcat.exceptions import ConfigError
from wardcat.llm.backends.base import BaseLLMBackend

#: A factory builds a backend from the ``llm_detector`` config sub-dict.
BackendFactory = Callable[[dict[str, Any]], BaseLLMBackend]


# ── Built-in backend factories (lazy imports — no heavy deps at module load) ──


def _make_ollama(cfg: dict[str, Any]) -> BaseLLMBackend:
    from wardcat.llm.backends.ollama import OllamaBackend
    from wardcat.llm.model_manager import ModelManager

    model = cfg.get("model", "llama3.2")
    backend = OllamaBackend(
        base_url=cfg.get("base_url", "http://localhost:11434"),
        model=model,
        allow_http=cfg.get("allow_http", False),
    )
    if cfg.get("auto_pull", False):
        ModelManager(backend).ensure_available(model, verbose=True)
    return backend


def _make_openai_compatible(cfg: dict[str, Any]) -> BaseLLMBackend:
    from wardcat.llm.backends.openai_compat import OpenAICompatBackend

    return OpenAICompatBackend(
        base_url=cfg.get("base_url", "http://localhost:11434"),
        model=cfg.get("model", "llama3.2"),
        api_key=cfg.get("api_key", ""),
        allow_http=cfg.get("allow_http", False),
    )


def _make_vllm(cfg: dict[str, Any]) -> BaseLLMBackend:
    from wardcat.llm.backends.vllm import DEFAULT_VLLM_BASE_URL, VLLMBackend

    return VLLMBackend(
        base_url=cfg.get("base_url", DEFAULT_VLLM_BASE_URL),
        model=cfg.get("model", ""),
        api_key=cfg.get("api_key", ""),
        allow_http=cfg.get("allow_http", False),
    )


def _make_transformers(cfg: dict[str, Any]) -> BaseLLMBackend:
    from wardcat.llm.backends.transformers_backend import TransformersBackend

    return TransformersBackend(
        model=cfg.get("model", "llama3.2"),
        device_map=cfg.get("device_map", "auto"),
        load_in_8bit=cfg.get("load_in_8bit", False),
        load_in_4bit=cfg.get("load_in_4bit", False),
    )


_BACKENDS: dict[str, BackendFactory] = {
    "ollama": _make_ollama,
    "openai_compatible": _make_openai_compatible,
    "vllm": _make_vllm,
    "transformers": _make_transformers,
}


def supported_backends() -> frozenset[str]:
    """The names of the built-in LLM backends."""
    return frozenset(_BACKENDS)


def create_backend(llm_cfg: dict[str, Any]) -> BaseLLMBackend:
    """Build the backend named by ``llm_cfg['backend']`` (default ``"ollama"``)."""
    name = llm_cfg.get("backend", "ollama")
    factory = _BACKENDS.get(name)
    if factory is None:
        raise ConfigError(f"Unknown LLM backend {name!r}. Supported backends: {sorted(_BACKENDS)}.")
    return factory(llm_cfg)

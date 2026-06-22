"""Registry of LLM backends — the extension point for new backends.

Each backend is registered as a factory ``(llm_cfg) -> BaseLLMBackend``. Adding a
backend (e.g. Azure OpenAI) needs no change to the core: register a factory and
ai-guard can build it. The built-in factories lazy-import their heavy deps
(torch/transformers), so importing this module stays cheap.

::

    from ai_guard import register_backend, BaseLLMBackend

    class MyBackend(BaseLLMBackend):
        ...

    register_backend("my_backend", lambda cfg: MyBackend(cfg["model"]))
    guard = AIGuard(salt="s").with_llm(backend="my_backend", model="...")
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_guard.exceptions import ConfigError
from ai_guard.llm.backends.base import BaseLLMBackend

#: A factory builds a backend from the ``llm_detector`` config sub-dict.
BackendFactory = Callable[[dict[str, Any]], BaseLLMBackend]

_REGISTRY: dict[str, BackendFactory] = {}


def register_backend(name: str, factory: BackendFactory) -> None:
    """Register (or override) an LLM backend factory under *name*.

    *factory* receives the ``llm_detector`` config dict (``base_url``, ``model``,
    ``api_key``, ``allow_http``, …) and returns a :class:`BaseLLMBackend`.
    """
    _REGISTRY[name] = factory


def registered_backends() -> frozenset[str]:
    """The names of all currently-registered backends (built-in + custom)."""
    return frozenset(_REGISTRY)


def create_backend(llm_cfg: dict[str, Any]) -> BaseLLMBackend:
    """Build the backend named by ``llm_cfg['backend']`` (default ``"ollama"``)."""
    name = llm_cfg.get("backend", "ollama")
    factory = _REGISTRY.get(name)
    if factory is None:
        raise ConfigError(
            f"Unknown LLM backend {name!r}. Registered backends: {sorted(_REGISTRY)}. "
            "Add one with ai_guard.register_backend(name, factory)."
        )
    return factory(llm_cfg)


# ── Built-in backends (lazy factories — no heavy imports at module load) ──────


def _make_ollama(cfg: dict[str, Any]) -> BaseLLMBackend:
    from ai_guard.llm.backends.ollama import OllamaBackend
    from ai_guard.llm.model_manager import ModelManager

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
    from ai_guard.llm.backends.openai_compat import OpenAICompatBackend

    return OpenAICompatBackend(
        base_url=cfg.get("base_url", "http://localhost:11434"),
        model=cfg.get("model", "llama3.2"),
        api_key=cfg.get("api_key", ""),
        allow_http=cfg.get("allow_http", False),
    )


def _make_transformers(cfg: dict[str, Any]) -> BaseLLMBackend:
    from ai_guard.llm.backends.transformers_backend import TransformersBackend

    return TransformersBackend(
        model=cfg.get("model", "llama3.2"),
        device_map=cfg.get("device_map", "auto"),
        load_in_8bit=cfg.get("load_in_8bit", False),
        load_in_4bit=cfg.get("load_in_4bit", False),
    )


register_backend("ollama", _make_ollama)
register_backend("openai_compatible", _make_openai_compatible)
register_backend("transformers", _make_transformers)

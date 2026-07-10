from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class Backend(str, Enum):
    """LLM backend types, as constants — for typo-proof selection.

    Pass these to :meth:`Wardcat.with_llm` instead of bare strings::

        from wardcat import Wardcat, Backend

        Wardcat(salt="s").with_llm(backend=Backend.OPENAI_COMPATIBLE, model="...")

    Each member *is* its string value (``Backend.OLLAMA == "ollama"``), so the
    plain string form is still accepted.
    """

    OLLAMA = "ollama"
    """Local Ollama service (supports model download)."""
    OPENAI_COMPATIBLE = "openai_compatible"
    """OpenAI-compatible HTTP API — LM Studio, LocalAI, LiteLLM, …"""
    VLLM = "vllm"
    """vLLM server (OpenAI-compatible API) — native chat, vLLM defaults."""
    TRANSFORMERS = "transformers"
    """In-process HuggingFace Transformers (no HTTP; loads the model locally)."""


@dataclass
class PullProgress:
    """Model download progress."""

    status: str
    completed: int = 0
    total: int = 0

    @property
    def percent(self) -> float:
        return (self.completed / self.total * 100) if self.total else 0.0


ProgressCallback = Callable[[PullProgress], None]


class BaseLLMBackend(ABC):
    """Interface that all LLM backends must implement."""

    @abstractmethod
    def complete(self, prompt: str, *, timeout: int = 60) -> str:
        """Send a prompt and return the completion text."""
        ...

    @abstractmethod
    def list_models(self) -> list[str]:
        """List available model names."""
        ...

    @abstractmethod
    def pull_model(
        self,
        model: str,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        """Download a model. The on_progress callback is called for progress updates."""
        ...

    async def complete_async(self, prompt: str, *, timeout: int = 60) -> str:
        """Async variant of :meth:`complete`.

        The default implementation wraps the synchronous :meth:`complete` in a
        thread pool so it does not block the event loop.  Backends that support
        native async I/O (e.g. via ``httpx.AsyncClient``) should override this.
        """
        return await asyncio.to_thread(self.complete, prompt, timeout=timeout)

    def complete_messages(self, messages: list[dict], *, timeout: int = 60) -> str:
        """Send a chat-formatted message list and return the assistant's reply.

        Default implementation concatenates messages into a plain-text prompt
        and delegates to :meth:`complete`.  Backends with native chat support
        (e.g. :class:`TransformersBackend`) should override this method.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            timeout:  Request timeout in seconds.
        """
        combined = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        return self.complete(combined, timeout=timeout)

    async def complete_messages_async(self, messages: list[dict], *, timeout: int = 60) -> str:
        """Async variant of :meth:`complete_messages`.

        The default implementation concatenates messages into a plain-text prompt
        and delegates to :meth:`complete_async`.  Backends with native async chat
        support should override this.
        """
        combined = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        return await self.complete_async(combined, timeout=timeout)

    def is_model_available(self, model: str) -> bool:
        """Return whether the model is ready in the service."""
        return model in self.list_models()

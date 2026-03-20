"""
Model management: downloading, listing, availability checking.
"""
from __future__ import annotations

import logging
import sys

from ai_guard.llm.backends.base import BaseLLMBackend, PullProgress

logger = logging.getLogger(__name__)


class ModelManager:
    """
    On-prem Llama model manager.

    Example::

        from ai_guard.llm.backends.ollama import OllamaBackend
        from ai_guard.llm.model_manager import ModelManager

        mgr = ModelManager(OllamaBackend())
        mgr.pull("llama3.2")
        print(mgr.list())
    """

    def __init__(self, backend: BaseLLMBackend) -> None:
        self.backend = backend

    def list(self) -> list[str]:
        """List models available in the service."""
        return self.backend.list_models()

    def is_available(self, model: str) -> bool:
        """Is the model ready to use in the service?"""
        return self.backend.is_model_available(model)

    def ensure_available(self, model: str, *, verbose: bool = True) -> bool:
        """
        Download the model if it is not available.

        :param model:   Model name, e.g. "llama3.1:8b"
        :param verbose: Print progress and notifications to the terminal
        :returns:       True — model is ready; False — download cancelled or error
        :raises NotImplementedError: If the backend does not support model downloading
        """
        if self.is_available(model):
            if verbose:
                logger.info("Model already available: %s", model)
            return True

        if verbose:
            logger.info("Model not found: %s", model)
            try:
                answer = input(f"  Download it? (~GB in size) [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = ""
            if answer not in ("y", "yes"):
                logger.info("Download cancelled by user")
                return False

        self.pull(model, verbose=verbose)
        return True

    def pull(self, model: str, *, verbose: bool = True) -> None:
        """
        Download a model.

        :param model:   Model name, e.g. "llama3.2" or "llama3.2:3b"
        :param verbose: Print progress to the terminal
        """
        if verbose:
            logger.info("Downloading model: %s", model)

        def _on_progress(p: PullProgress) -> None:
            if not verbose:
                return
            if p.total:
                bar_len  = 30
                filled   = int(bar_len * p.completed / p.total)
                bar      = "█" * filled + "░" * (bar_len - filled)
                print(f"\r  [{bar}] {p.percent:5.1f}%  {p.status:<20}", end="", flush=True)
            else:
                print(f"\r  {p.status:<40}", end="", flush=True)

        self.backend.pull_model(model, on_progress=_on_progress)

        if verbose:
            logger.info("Model ready: %s", model)

"""
Model management: downloading, listing, availability checking.
"""

from __future__ import annotations

import logging

from wardcat.llm.backends.base import BaseLLMBackend, ProgressCallback, PullProgress

logger = logging.getLogger(__name__)


class ModelManager:
    """
    On-prem Llama model manager.

    Example::

        from wardcat.llm.backends.ollama import OllamaBackend
        from wardcat.llm.model_manager import ModelManager

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
                answer = input("  Download it? (~GB in size) [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = ""
            if answer not in ("y", "yes"):
                logger.info("Download cancelled by user")
                return False

        self.pull(model, verbose=verbose)
        return True

    def pull(
        self,
        model: str,
        *,
        verbose: bool = True,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        """
        Download a model.

        :param model:       Model name, e.g. "llama3.2" or "llama3.2:3b"
        :param verbose:     Log start/finish and, when no ``on_progress`` is given,
                            render a terminal progress bar.
        :param on_progress: Callback invoked with each :class:`PullProgress`. Pass
                            your own to integrate with a UI/logger instead of the
                            default terminal bar (a library should not print for
                            you unless you ask).
        """
        logger.info("Downloading model: %s", model)

        callback = on_progress if on_progress is not None else self._terminal_bar(verbose)
        self.backend.pull_model(model, on_progress=callback)

        logger.info("Model ready: %s", model)

    @staticmethod
    def _terminal_bar(verbose: bool) -> ProgressCallback | None:
        """Default progress renderer — a terminal bar, only when ``verbose``."""
        if not verbose:
            return None

        def _render(p: PullProgress) -> None:
            if p.total:
                bar_len = 30
                filled = int(bar_len * p.completed / p.total)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"\r  [{bar}] {p.percent:5.1f}%  {p.status:<20}", end="", flush=True)
            else:
                print(f"\r  {p.status:<40}", end="", flush=True)

        return _render

"""
Model setup flow integration tests.

Tests the Wardcat ``with_llm(auto_pull=...)`` flow with a mock Ollama backend.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ── Helper ───────────────────────────────────────────────────────────────────


def _mock_ollama(available: list[str]):
    """Mock OllamaBackend; no real HTTP calls are made."""
    from wardcat.llm.backends.ollama import OllamaBackend

    backend = MagicMock(spec=OllamaBackend)
    backend.base_url = "http://localhost:11434"
    backend.model = available[0] if available else "llama3.1:8b"
    backend.list_models.return_value = available
    backend.is_model_available.side_effect = lambda m: m in available
    backend.pull_model.return_value = None
    backend.complete.return_value = "[]"
    backend.complete_messages.return_value = "[]"
    return backend


# ── Wardcat auto_pull ───────────────────────────────────────────────────────


class TestWardcatAutoPull:
    def test_auto_pull_true_model_missing_user_confirms(self):
        """
        auto_pull=True + model missing + user says 'y' → pull is called.
        """
        mock_backend = _mock_ollama([])

        with (
            patch("wardcat.llm.backends.ollama.OllamaBackend", return_value=mock_backend),
            patch("builtins.input", return_value="y"),
        ):
            from wardcat import Wardcat

            Wardcat(use_ner=False).with_llm(model="llama3.1:8b", auto_pull=True)

        mock_backend.pull_model.assert_called_once()
        args, _ = mock_backend.pull_model.call_args
        assert args[0] == "llama3.1:8b"

    def test_auto_pull_true_model_missing_user_declines(self):
        """
        auto_pull=True + model missing + user says 'n' → pull not called,
        guard is still created (LLM may not work but does not crash).
        """
        mock_backend = _mock_ollama([])

        with (
            patch("wardcat.llm.backends.ollama.OllamaBackend", return_value=mock_backend),
            patch("builtins.input", return_value="n"),
        ):
            from wardcat import Wardcat

            Wardcat(use_ner=False).with_llm(model="llama3.1:8b", auto_pull=True)

        mock_backend.pull_model.assert_not_called()

    def test_auto_pull_false_no_pull_check(self):
        """
        auto_pull=False (default) → ensure_available is never called.
        """
        mock_backend = _mock_ollama(["llama3.1:8b"])

        with patch("wardcat.llm.backends.ollama.OllamaBackend", return_value=mock_backend):
            from wardcat import Wardcat

            Wardcat(use_ner=False).with_llm(model="llama3.1:8b", auto_pull=False)

        mock_backend.is_model_available.assert_not_called()

    def test_auto_pull_true_model_already_present_no_pull(self):
        """Model already present → pull not called."""
        mock_backend = _mock_ollama(["llama3.1:8b"])

        with patch("wardcat.llm.backends.ollama.OllamaBackend", return_value=mock_backend):
            from wardcat import Wardcat

            Wardcat(use_ner=False).with_llm(model="llama3.1:8b", auto_pull=True)

        mock_backend.pull_model.assert_not_called()

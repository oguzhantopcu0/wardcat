"""
Model setup flow integration tests.

Tests LLMGuard auto_pull + CLI models setup + models list --recommended
flows with a mock Ollama backend.
"""
from __future__ import annotations

import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from ai_guard.llm.backends.base import BaseLLMBackend
from ai_guard.llm.model_catalog import CATALOG, recommended
from ai_guard.llm.model_manager import ModelManager


# ── Helper ───────────────────────────────────────────────────────────────────

def _mock_ollama(available: list[str]):
    """Mock OllamaBackend; no real HTTP calls are made."""
    from ai_guard.llm.backends.ollama import OllamaBackend
    backend = MagicMock(spec=OllamaBackend)
    backend.base_url = "http://localhost:11434"
    backend.model    = available[0] if available else "llama3.1:8b"
    backend.list_models.return_value       = available
    backend.is_model_available.side_effect = lambda m: m in available
    backend.pull_model.return_value        = None
    backend.complete.return_value          = "[]"
    backend.complete_messages.return_value = "[]"
    return backend


# ── LLMGuard auto_pull ───────────────────────────────────────────────────────

class TestLLMGuardAutoPull:
    def test_auto_pull_true_model_missing_user_confirms(self):
        """
        auto_pull=True + model missing + user says 'y' → pull is called.
        """
        mock_backend = _mock_ollama([])

        with (
            patch("ai_guard.llm.backends.ollama.OllamaBackend",
                  return_value=mock_backend),
            patch("builtins.input", return_value="y"),
        ):
            from ai_guard import LLMGuard
            guard = LLMGuard(
                use_ner=False,
                use_llm=True,
                llm_model="llama3.1:8b",
                auto_pull=True,
            )

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
            patch("ai_guard.llm.backends.ollama.OllamaBackend",
                  return_value=mock_backend),
            patch("builtins.input", return_value="n"),
        ):
            from ai_guard import LLMGuard
            guard = LLMGuard(
                use_ner=False,
                use_llm=True,
                llm_model="llama3.1:8b",
                auto_pull=True,
            )

        mock_backend.pull_model.assert_not_called()

    def test_auto_pull_false_no_pull_check(self):
        """
        auto_pull=False (default) → ensure_available is never called.
        """
        mock_backend = _mock_ollama(["llama3.1:8b"])

        with patch("ai_guard.llm.backends.ollama.OllamaBackend",
                   return_value=mock_backend):
            from ai_guard import LLMGuard
            guard = LLMGuard(
                use_ner=False,
                use_llm=True,
                llm_model="llama3.1:8b",
                auto_pull=False,
            )

        mock_backend.is_model_available.assert_not_called()

    def test_auto_pull_true_model_already_present_no_pull(self):
        """Model already present → pull not called."""
        mock_backend = _mock_ollama(["llama3.1:8b"])

        with patch("ai_guard.llm.backends.ollama.OllamaBackend",
                   return_value=mock_backend):
            from ai_guard import LLMGuard
            guard = LLMGuard(
                use_ner=False,
                use_llm=True,
                llm_model="llama3.1:8b",
                auto_pull=True,
            )

        mock_backend.pull_model.assert_not_called()


# ── CLI models list --recommended ────────────────────────────────────────────

class TestCLIModelsList:
    def _run_cli(self, *args):
        """Run CLI directly instead of subprocess; capture stdout."""
        from ai_guard.__main__ import _build_parser, cmd_models
        parser = _build_parser()
        parsed = parser.parse_args(["models", "list", *args])
        buf = StringIO()
        with patch("sys.stdout", buf):
            cmd_models(parsed)
        return buf.getvalue()

    def test_recommended_flag_prints_catalog(self):
        output = self._run_cli("--recommended")
        # All models in the catalog should be in the output
        for m in CATALOG:
            assert m.name in output

    def test_recommended_flag_marks_recommended(self):
        output = self._run_cli("--recommended")
        assert "recommended" in output

    def test_recommended_flag_shows_vram(self):
        output = self._run_cli("--recommended")
        assert "GB" in output

    def test_without_recommended_hits_ollama(self):
        """Without --recommended, a real Ollama backend call is made."""
        mock_backend = _mock_ollama(["llama3.1:8b", "mistral:7b"])
        with patch("ai_guard.llm.backends.ollama.OllamaBackend",
                   return_value=mock_backend):
            output = self._run_cli()
        assert "llama3.1:8b" in output
        assert "mistral:7b"  in output


# ── CLI models setup ──────────────────────────────────────────────────────────

class TestCLIModelsSetup:
    def _run_setup(self, user_input: str, available: list[str] | None = None):
        from ai_guard.__main__ import _build_parser, cmd_models
        parser = _build_parser()
        parsed = parser.parse_args(["models", "setup"])

        mock_backend = _mock_ollama(available or [])
        buf = StringIO()

        with (
            patch("ai_guard.llm.backends.ollama.OllamaBackend",
                  return_value=mock_backend),
            patch("builtins.input", return_value=user_input),
            patch("sys.stdout", buf),
        ):
            cmd_models(parsed)

        return buf.getvalue(), mock_backend

    def test_setup_shows_catalog(self):
        output, _ = self._run_setup("h")  # cancel
        for m in CATALOG:
            assert m.name in output

    def test_setup_empty_input_selects_first(self):
        """Empty input → first model (llama3.1:8b) should be selected."""
        # Two-stage input: model no (empty → 1st model), then download confirmation (y)
        mock_backend = _mock_ollama([])
        inputs = iter(["", "y"])  # model selection: empty, download: yes

        from ai_guard.__main__ import _build_parser, cmd_models
        parser = _build_parser()
        parsed = parser.parse_args(["models", "setup"])

        with (
            patch("ai_guard.llm.backends.ollama.OllamaBackend",
                  return_value=mock_backend),
            patch("builtins.input", side_effect=inputs),
            patch("sys.stdout", StringIO()),
        ):
            cmd_models(parsed)

        mock_backend.pull_model.assert_called_once()
        args, _ = mock_backend.pull_model.call_args
        assert args[0] == CATALOG[0].name

    def test_setup_valid_number_selects_correct(self):
        """Input '2' → should select 2nd catalog model."""
        # First input: model no, second input: download confirmation
        inputs = iter(["2", "y"])
        mock_backend = _mock_ollama([])

        from ai_guard.__main__ import _build_parser, cmd_models
        parser = _build_parser()
        parsed = parser.parse_args(["models", "setup"])

        with (
            patch("ai_guard.llm.backends.ollama.OllamaBackend",
                  return_value=mock_backend),
            patch("builtins.input", side_effect=inputs),
            patch("sys.stdout", StringIO()),
        ):
            cmd_models(parsed)

        mock_backend.pull_model.assert_called_once()
        args, _ = mock_backend.pull_model.call_args
        assert args[0] == CATALOG[1].name

    def test_setup_invalid_number_cancels(self):
        output, backend = self._run_setup("99")
        backend.pull_model.assert_not_called()
        assert "Invalid" in output or "cancelled" in output.lower()

    def test_setup_non_interactive_uses_default(self):
        """--non-interactive → select default model without prompting the user."""
        from ai_guard.__main__ import _build_parser, cmd_models
        parser = _build_parser()
        parsed = parser.parse_args(["models", "setup", "--non-interactive"])

        mock_backend = _mock_ollama(["llama3.1:8b"])
        buf = StringIO()

        with (
            patch("ai_guard.llm.backends.ollama.OllamaBackend",
                  return_value=mock_backend),
            patch("builtins.input", side_effect=RuntimeError("input called!")),
            patch("sys.stdout", buf),
        ):
            cmd_models(parsed)  # input() should not be called

        # Model already present → pull not called, no error either
        mock_backend.pull_model.assert_not_called()

    def test_setup_model_already_installed_no_pull(self):
        _, backend = self._run_setup("", available=["llama3.1:8b"])
        backend.pull_model.assert_not_called()

    def test_setup_shows_usage_after_install(self):
        # Model already present → only model no input needed, no download confirmation asked
        mock_backend = _mock_ollama(["llama3.1:8b"])
        from ai_guard.__main__ import _build_parser, cmd_models
        parser = _build_parser()
        parsed = parser.parse_args(["models", "setup"])
        buf = StringIO()
        with (
            patch("ai_guard.llm.backends.ollama.OllamaBackend",
                  return_value=mock_backend),
            patch("builtins.input", return_value=""),  # empty → 1st model
            patch("sys.stdout", buf),
        ):
            cmd_models(parsed)
        output = buf.getvalue()
        # Model present → usage instructions should be shown
        assert "llm-model" in output or "use_llm" in output


# ── CLI scan --auto-pull flag ─────────────────────────────────────────────────

class TestCLIScanAutoPull:
    def test_auto_pull_flag_passed_to_guard(self):
        """--auto-pull flag should result in LLMGuard(auto_pull=True) being called."""
        from ai_guard.__main__ import _build_parser
        parser = _build_parser()
        args   = parser.parse_args([
            "scan", "--text", "test",
            "--llm", "--llm-model", "llama3.1:8b", "--auto-pull",
        ])
        assert args.auto_pull is True

    def test_no_auto_pull_flag_default_false(self):
        from ai_guard.__main__ import _build_parser
        parser = _build_parser()
        args   = parser.parse_args(["scan", "--text", "test"])
        assert args.auto_pull is False

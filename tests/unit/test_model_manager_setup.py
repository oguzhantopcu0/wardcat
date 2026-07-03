"""
ModelManager.ensure_available() unit tests.

No real Ollama connection is made; the backend is mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from wardcat.llm.backends.base import BaseLLMBackend, PullProgress
from wardcat.llm.model_manager import ModelManager


def _mgr(available: list[str]) -> ModelManager:
    backend = MagicMock(spec=BaseLLMBackend)
    backend.list_models.return_value = available
    backend.is_model_available.side_effect = lambda m: m in available
    backend.pull_model.return_value = None
    return ModelManager(backend)


class TestEnsureAvailable:
    def test_already_available_returns_true(self):
        mgr = _mgr(["llama3.1:8b"])
        assert mgr.ensure_available("llama3.1:8b", verbose=False) is True

    def test_already_available_no_pull(self):
        mgr = _mgr(["llama3.1:8b"])
        mgr.ensure_available("llama3.1:8b", verbose=False)
        mgr.backend.pull_model.assert_not_called()

    def test_not_available_user_confirms_pulls(self):
        mgr = _mgr([])
        with patch("builtins.input", return_value="y"):
            result = mgr.ensure_available("llama3.1:8b", verbose=True)
        assert result is True
        mgr.backend.pull_model.assert_called_once()
        args, _ = mgr.backend.pull_model.call_args
        assert args[0] == "llama3.1:8b"

    def test_not_available_user_declines_returns_false(self):
        mgr = _mgr([])
        with patch("builtins.input", return_value="h"):
            result = mgr.ensure_available("llama3.1:8b", verbose=True)
        assert result is False
        mgr.backend.pull_model.assert_not_called()

    def test_not_available_user_empty_input_declines(self):
        mgr = _mgr([])
        with patch("builtins.input", return_value=""):
            result = mgr.ensure_available("llama3.1:8b", verbose=True)
        assert result is False

    def test_not_available_eof_declines(self):
        """No TTY (CI/pipe) → EOFError → cancelled."""
        mgr = _mgr([])
        with patch("builtins.input", side_effect=EOFError):
            result = mgr.ensure_available("llama3.1:8b", verbose=True)
        assert result is False
        mgr.backend.pull_model.assert_not_called()

    def test_verbose_false_no_input_prompt(self):
        """verbose=False → input() should not be called, otherwise error."""
        mgr = _mgr(["llama3.1:8b"])
        # If input is called, RuntimeError (without patch, real stdin would be used)
        with patch("builtins.input", side_effect=RuntimeError("input called!")):
            result = mgr.ensure_available("llama3.1:8b", verbose=False)
        assert result is True  # already available, no input needed

    @pytest.mark.parametrize("answer", ["y", "yes", "Y", "YES"])
    def test_affirmative_answers_trigger_pull(self, answer):
        mgr = _mgr([])
        with patch("builtins.input", return_value=answer):
            result = mgr.ensure_available("llama3.1:8b", verbose=True)
        assert result is True
        mgr.backend.pull_model.assert_called_once()

    @pytest.mark.parametrize("answer", ["h", "n", "no", "H", "x", "nope"])
    def test_negative_answers_cancel(self, answer):
        mgr = _mgr([])
        with patch("builtins.input", return_value=answer):
            result = mgr.ensure_available("llama3.1:8b", verbose=True)
        assert result is False


class TestPullStillWorks:
    """Adding ensure_available should not have broken pull()."""

    def test_pull_calls_backend(self):
        mgr = _mgr([])
        mgr.pull("llama3.1:8b", verbose=False)
        mgr.backend.pull_model.assert_called_once()

    def test_pull_verbose_prints(self, capsys, caplog):
        import logging

        backend = MagicMock(spec=BaseLLMBackend)

        def fake_pull(model, *, on_progress=None):
            if on_progress:
                on_progress(PullProgress("downloading", 50, 100))
                on_progress(PullProgress("success", 100, 100))

        backend.pull_model.side_effect = fake_pull
        with caplog.at_level(logging.INFO, logger="wardcat.llm.model_manager"):
            ModelManager(backend).pull("llama3.1:8b", verbose=True)
        # Model name appears in log messages; progress bar goes to stdout
        assert "llama3.1:8b" in caplog.text
        assert capsys.readouterr().out != ""  # progress bar still printed

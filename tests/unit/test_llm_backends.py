"""
LLM backend units: backend creation, error handling,
model_manager interface — no real HTTP calls are made.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call

import pytest

from ai_guard.llm.backends.base import BaseLLMBackend, PullProgress
from ai_guard.llm.backends.ollama import OllamaBackend
from ai_guard.llm.backends.openai_compat import OpenAICompatBackend
from ai_guard.llm.model_manager import ModelManager


# ═══════════════════════════════════════════════════════════════════════════
# OllamaBackend
# ═══════════════════════════════════════════════════════════════════════════

class TestOllamaBackend:
    def test_default_url_and_model(self):
        b = OllamaBackend()
        assert b.base_url == "http://localhost:11434"
        assert b.model    == "llama3.2"

    def test_trailing_slash_stripped(self):
        b = OllamaBackend(base_url="http://localhost:11434/")
        assert b.base_url == "http://localhost:11434"

    def test_complete_sends_correct_payload(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "test output"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            b = OllamaBackend(model="llama3.2:3b")
            result = b.complete("merhaba")

        assert result == "test output"
        payload = mock_post.call_args[1]["json"]
        assert payload["model"]  == "llama3.2:3b"
        assert payload["prompt"] == "merhaba"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0

    def test_complete_connection_error_raises(self):
        import httpx
        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(ConnectionError, match="Ollama"):
                OllamaBackend().complete("test")

    def test_list_models_parses_response(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [{"name": "llama3.2"}, {"name": "mistral"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            models = OllamaBackend().list_models()

        assert "llama3.2" in models
        assert "mistral"  in models

    def test_list_models_empty_server(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            assert OllamaBackend().list_models() == []

    def test_is_model_available_true(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": [{"name": "llama3.2"}]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            assert OllamaBackend().is_model_available("llama3.2") is True

    def test_is_model_available_false(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            assert OllamaBackend().is_model_available("llama3.2") is False

    def test_pull_model_calls_progress(self):
        progress_lines = [
            json.dumps({"status": "pulling manifest"}),
            json.dumps({"status": "downloading", "completed": 50, "total": 100}),
            json.dumps({"status": "success"}),
        ]
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__  = MagicMock(return_value=False)
        mock_stream.iter_lines = MagicMock(return_value=iter(progress_lines))
        mock_stream.raise_for_status = MagicMock()

        collected: list[PullProgress] = []

        with patch("httpx.stream", return_value=mock_stream):
            OllamaBackend().pull_model("llama3.2", on_progress=collected.append)

        assert len(collected) == 3
        assert collected[1].completed == 50
        assert collected[1].total     == 100
        assert abs(collected[1].percent - 50.0) < 0.1


# ═══════════════════════════════════════════════════════════════════════════
# OpenAICompatBackend
# ═══════════════════════════════════════════════════════════════════════════

class TestOpenAICompatBackend:
    def test_complete_sends_chat_format(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "result"}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            b = OpenAICompatBackend(base_url="http://vllm:8000/v1", model="llama3")
            result = b.complete("prompt")

        assert result == "result"
        payload = mock_post.call_args[1]["json"]
        assert payload["model"]      == "llama3"
        assert payload["messages"][0]["role"]    == "user"
        assert payload["messages"][0]["content"] == "prompt"
        assert payload["temperature"] == 0

    def test_api_key_added_to_header(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": ""}}]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            b = OpenAICompatBackend(base_url="http://x", model="m", api_key="secret")
            b.complete("test")

        headers = mock_post.call_args[1]["headers"]
        assert headers.get("Authorization") == "Bearer secret"

    def test_no_api_key_no_auth_header(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": ""}}]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            b = OpenAICompatBackend(base_url="http://x", model="m")
            b.complete("test")

        headers = mock_post.call_args[1].get("headers", {})
        assert "Authorization" not in headers

    def test_pull_model_raises_not_implemented(self):
        b = OpenAICompatBackend(base_url="http://x", model="m")
        with pytest.raises(NotImplementedError):
            b.pull_model("any-model")

    def test_list_models_parses_openai_format(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"id": "llama3"}, {"id": "mistral"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            b = OpenAICompatBackend(base_url="http://x", model="m")
            models = b.list_models()

        assert "llama3"  in models
        assert "mistral" in models


# ═══════════════════════════════════════════════════════════════════════════
# ModelManager
# ═══════════════════════════════════════════════════════════════════════════

class TestModelManager:
    def _mgr(self, models: list[str] | None = None) -> ModelManager:
        backend = MagicMock(spec=BaseLLMBackend)
        backend.list_models.return_value = models or []
        backend.is_model_available.side_effect = lambda m: m in (models or [])
        return ModelManager(backend)

    def test_list_delegates_to_backend(self):
        mgr = self._mgr(["llama3.2", "mistral"])
        assert "llama3.2" in mgr.list()
        assert "mistral"  in mgr.list()

    def test_is_available_true(self):
        mgr = self._mgr(["llama3.2"])
        assert mgr.is_available("llama3.2") is True

    def test_is_available_false(self):
        mgr = self._mgr([])
        assert mgr.is_available("llama3.2") is False

    def test_pull_calls_backend_pull(self):
        backend = MagicMock(spec=BaseLLMBackend)
        mgr = ModelManager(backend)
        mgr.pull("llama3.2", verbose=False)
        backend.pull_model.assert_called_once()
        args, kwargs = backend.pull_model.call_args
        assert args[0] == "llama3.2"

    def test_pull_verbose_false_no_print(self, capsys):
        backend = MagicMock(spec=BaseLLMBackend)
        ModelManager(backend).pull("llama3.2", verbose=False)
        assert capsys.readouterr().out == ""

    def test_pull_verbose_true_prints_progress(self, capsys):
        backend = MagicMock(spec=BaseLLMBackend)

        def fake_pull(model, *, on_progress=None):
            if on_progress:
                on_progress(PullProgress("downloading", 50, 100))
                on_progress(PullProgress("success", 100, 100))

        backend.pull_model.side_effect = fake_pull
        ModelManager(backend).pull("llama3.2", verbose=True)
        out = capsys.readouterr().out
        assert "llama3.2" in out


# ═══════════════════════════════════════════════════════════════════════════
# httpx ImportError
# ═══════════════════════════════════════════════════════════════════════════

class TestHttpxImportError:
    def test_ollama_httpx_missing_raises(self):
        import sys
        from unittest.mock import patch
        with patch.dict(sys.modules, {"httpx": None}):
            from ai_guard.llm.backends import ollama as _ollama_mod
            import importlib
            # _httpx() is called on use, not import — call complete() to trigger it
            backend = OllamaBackend.__new__(OllamaBackend)
            backend.base_url = "http://localhost:11434"
            backend.model = "llama3.2"
            # Patch _httpx directly
            with patch("ai_guard.llm.backends.ollama._httpx", side_effect=ImportError("httpx missing")):
                with pytest.raises(ImportError, match="httpx"):
                    backend.complete("test")

    def test_openai_compat_httpx_missing_raises(self):
        with patch("ai_guard.llm.backends.openai_compat._httpx", side_effect=ImportError("httpx missing")):
            backend = OpenAICompatBackend.__new__(OpenAICompatBackend)
            backend.base_url = "http://x"
            backend.model = "m"
            backend._headers = {}
            with pytest.raises(ImportError, match="httpx"):
                backend.complete("test")


class TestOllamaConnectErrors:
    def test_list_models_connect_error(self):
        import httpx
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(ConnectionError, match="Ollama"):
                OllamaBackend().list_models()

    def test_pull_model_connect_error(self):
        import httpx
        with patch("httpx.stream", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(ConnectionError, match="Ollama"):
                OllamaBackend().pull_model("llama3.2")

    def test_pull_model_skips_empty_lines(self):
        """Empty lines in the streaming response should be silently skipped."""
        lines = ["", '{"status": "pulling"}', "", '{"status": "success"}']
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.iter_lines = MagicMock(return_value=iter(lines))
        mock_stream.raise_for_status = MagicMock()

        collected = []
        with patch("httpx.stream", return_value=mock_stream):
            OllamaBackend().pull_model("llama3.2", on_progress=collected.append)

        # Only non-empty lines produce progress events
        assert len(collected) == 2

    def test_complete_messages_delegates(self):
        """complete_messages delegates to complete() via base class."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "result"}  # /api/generate format
        mock_response.raise_for_status = MagicMock()
        with patch("httpx.post", return_value=mock_response):
            backend = OllamaBackend(model="llama3.2")
            result = backend.complete_messages([{"role": "user", "content": "hello"}])
        assert result == "result"


class TestOpenAICompatConnectErrors:
    def test_complete_connect_error(self):
        import httpx
        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(ConnectionError, match="LLM service"):
                OpenAICompatBackend(base_url="http://x", model="m").complete("test")

    def test_list_models_connect_error(self):
        import httpx
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(ConnectionError, match="LLM service"):
                OpenAICompatBackend(base_url="http://x", model="m").list_models()

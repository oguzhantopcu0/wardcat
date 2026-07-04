"""
vLLM backend tests.

VLLMBackend subclasses OpenAICompatBackend (vLLM serves an OpenAI-compatible
API) and adds vLLM defaults plus a native chat path — ``complete_messages``
posts the real messages array instead of the base class's flattened prompt.
All HTTP is mocked; no real vLLM server is contacted.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wardcat.llm.backends.base import Backend
from wardcat.llm.backends.registry import create_backend, registered_backends
from wardcat.llm.backends.vllm import DEFAULT_VLLM_BASE_URL, VLLMBackend


def _mock_post(content: str = "ok") -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    resp.raise_for_status = MagicMock()
    return resp


# ── Defaults & wiring ────────────────────────────────────────────────────────


class TestVLLMDefaults:
    def test_default_base_url_is_vllm_port(self):
        b = VLLMBackend()
        assert b.base_url == DEFAULT_VLLM_BASE_URL == "http://localhost:8000/v1"

    def test_backend_enum_has_vllm(self):
        assert Backend.VLLM == "vllm"

    def test_registered_as_vllm(self):
        assert "vllm" in registered_backends()

    def test_create_backend_builds_vllm(self):
        backend = create_backend(
            {"backend": "vllm", "model": "my-model", "base_url": "http://localhost:8000/v1"}
        )
        assert isinstance(backend, VLLMBackend)
        assert backend.model == "my-model"

    def test_pull_model_raises_not_implemented(self):
        # Model management is server-side (inherited behaviour).
        with pytest.raises(NotImplementedError):
            VLLMBackend(model="m").pull_model("any")


# ── Native chat: complete_messages posts the real messages array ─────────────


class TestVLLMNativeChat:
    def test_complete_messages_preserves_roles(self):
        messages = [
            {"role": "system", "content": "You detect PII."},
            {"role": "user", "content": "email a@b.com"},
        ]
        with patch("httpx.post", return_value=_mock_post("found")) as mock_post:
            result = VLLMBackend(model="llama3").complete_messages(messages)

        assert result == "found"
        payload = mock_post.call_args[1]["json"]
        # The full messages array is sent verbatim — not flattened to one prompt.
        assert payload["messages"] == messages
        assert payload["model"] == "llama3"
        assert payload["temperature"] == 0
        assert mock_post.call_args[0][0].endswith("/chat/completions")

    def test_api_key_becomes_bearer_header(self):
        with patch("httpx.post", return_value=_mock_post()) as mock_post:
            VLLMBackend(model="m", api_key="secret").complete_messages(
                [{"role": "user", "content": "x"}]
            )
        assert mock_post.call_args[1]["headers"]["Authorization"] == "Bearer secret"

    def test_no_api_key_no_auth_header(self):
        with patch("httpx.post", return_value=_mock_post()) as mock_post:
            VLLMBackend(model="m").complete_messages([{"role": "user", "content": "x"}])
        assert "Authorization" not in mock_post.call_args[1].get("headers", {})

    def test_connect_error_raises_vllm_connectionerror(self):
        import httpx

        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(ConnectionError, match="vLLM service"):
                VLLMBackend(model="m").complete_messages([{"role": "user", "content": "x"}])

    def test_bad_response_format_raises(self):
        resp = MagicMock()
        resp.json.return_value = {"unexpected": "shape"}
        resp.raise_for_status = MagicMock()
        with patch("httpx.post", return_value=resp):
            with pytest.raises(ConnectionError, match="Unexpected response"):
                VLLMBackend(model="m").complete_messages([{"role": "user", "content": "x"}])


# ── Native async chat ────────────────────────────────────────────────────────


class TestVLLMNativeChatAsync:
    def _mock_client(self, content: str = "ok") -> MagicMock:
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.post = AsyncMock(return_value=_mock_post(content))
        return client

    def test_complete_messages_async_sends_array(self):
        client = self._mock_client("async-found")
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
        ]
        with patch("httpx.AsyncClient", return_value=client):
            result = asyncio.run(VLLMBackend(model="m").complete_messages_async(messages))
        assert result == "async-found"
        assert client.post.call_args[1]["json"]["messages"] == messages

    def test_complete_messages_async_connect_error(self):
        import httpx

        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        with patch("httpx.AsyncClient", return_value=client):
            with pytest.raises(ConnectionError, match="vLLM service"):
                asyncio.run(
                    VLLMBackend(model="m").complete_messages_async(
                        [{"role": "user", "content": "x"}]
                    )
                )


# ── Inherited transport still works (complete / list_models) ─────────────────


class TestVLLMInherited:
    def test_complete_single_prompt_still_works(self):
        with patch("httpx.post", return_value=_mock_post("single")) as mock_post:
            result = VLLMBackend(model="m").complete("hello")
        assert result == "single"
        # Inherited complete() wraps the prompt as a single user message.
        assert mock_post.call_args[1]["json"]["messages"] == [{"role": "user", "content": "hello"}]

    def test_list_models_parses_openai_format(self):
        resp = MagicMock()
        resp.json.return_value = {"data": [{"id": "llama3"}, {"id": "mistral"}]}
        resp.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=resp):
            models = VLLMBackend(model="m").list_models()
        assert models == ["llama3", "mistral"]

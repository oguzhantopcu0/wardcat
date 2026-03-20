from __future__ import annotations

import logging

from ai_guard.llm.backends.base import BaseLLMBackend, ProgressCallback

logger = logging.getLogger(__name__)


def _warn_if_http(url: str) -> None:
    """Warn if an unencrypted HTTP connection is being used."""
    if url.startswith("http://"):
        logger.warning(
            "LLM backend is using HTTP: %s — PII will be transmitted unencrypted. "
            "Use HTTPS in production (e.g. nginx/Caddy reverse proxy).",
            url,
        )


def _httpx():
    try:
        import httpx
        return httpx
    except ImportError:
        raise ImportError(
            "'httpx' is required for the LLM detector. "
            "Install with: uv add 'ai-guard[llm]'"
        )


class OpenAICompatBackend(BaseLLMBackend):
    """
    OpenAI-compatible REST API backend.

    Supports services like vLLM, LM Studio, LocalAI, and LiteLLM.
    Model downloading is not available via the API for these backends;
    it must be managed server-side.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        _warn_if_http(self.base_url)

    def complete(self, prompt: str, *, timeout: int = 60) -> str:
        httpx = _httpx()
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except httpx.ConnectError:
            raise ConnectionError(
                f"Could not connect to LLM service: {self.base_url}"
            )

    def list_models(self) -> list[str]:
        httpx = _httpx()
        try:
            response = httpx.get(
                f"{self.base_url}/models",
                headers=self._headers,
                timeout=10,
            )
            response.raise_for_status()
            return [m["id"] for m in response.json().get("data", [])]
        except httpx.ConnectError:
            raise ConnectionError(f"Could not connect to LLM service: {self.base_url}")

    def pull_model(
        self,
        model: str,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        raise NotImplementedError(
            "Model downloading is not supported in the OpenAI-compatible backend. "
            "Manage the model server-side (vLLM, LM Studio, etc.)."
        )

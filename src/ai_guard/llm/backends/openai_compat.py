from __future__ import annotations

import logging
import os

from ai_guard.llm.backends.base import BaseLLMBackend, ProgressCallback

logger = logging.getLogger(__name__)

_LOCALHOST_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


def _warn_if_http(url: str) -> None:
    """Enforce HTTPS for remote hosts; warn only for localhost.

    See :func:`ai_guard.llm.backends.ollama._warn_if_http` for policy details.
    """
    if not url.startswith("http://"):
        return
    if os.environ.get("LLMGUARD_ALLOW_HTTP", "").lower() in ("1", "true", "yes"):
        return
    host = url[len("http://"):].split("/")[0].split(":")[0]
    if host in _LOCALHOST_HOSTS:
        logger.warning(
            "LLM backend is using HTTP: %s — PII will be transmitted unencrypted. "
            "Use HTTPS in production (e.g. nginx/Caddy reverse proxy).",
            url,
        )
    else:
        raise ValueError(
            f"LLM backend HTTP connection to remote host is not allowed: {url}\n"
            "PII would be transmitted unencrypted over the network.\n"
            "Use HTTPS, or set LLMGUARD_ALLOW_HTTP=true to override (not recommended)."
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
            data = response.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as exc:
                raise ConnectionError(
                    f"Unexpected response format from LLM service (missing expected keys): {exc}"
                )
        except httpx.ConnectError:
            raise ConnectionError(
                f"Could not connect to LLM service: {self.base_url}"
            )

    async def complete_async(self, prompt: str, *, timeout: int = 60) -> str:
        """Native async variant using ``httpx.AsyncClient``."""
        httpx = _httpx()
        try:
            async with httpx.AsyncClient(headers=self._headers) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                    },
                    timeout=timeout,
                )
                response.raise_for_status()
                data = response.json()
                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError) as exc:
                    raise ConnectionError(
                        f"Unexpected response format from LLM service: {exc}"
                    )
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

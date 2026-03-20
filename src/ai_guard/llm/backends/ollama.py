from __future__ import annotations

import json
import logging
import os

from ai_guard.llm.backends.base import BaseLLMBackend, ProgressCallback, PullProgress

logger = logging.getLogger(__name__)

_LOCALHOST_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


def _warn_if_http(url: str) -> None:
    """Enforce HTTPS for remote hosts; warn only for localhost.

    For remote HTTP connections PII would traverse the network in plaintext,
    so a ``ValueError`` is raised unless the ``LLMGUARD_ALLOW_HTTP=true``
    environment variable is set.  Localhost connections are only warned
    (they may still be intercepted by processes on the same host, but the
    risk is considerably lower than a remote plaintext hop).
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


class OllamaBackend(BaseLLMBackend):
    """
    Ollama REST API backend.

    Default: http://localhost:11434
    Ollama runs models locally and provides download management.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        _warn_if_http(self.base_url)

    def complete(self, prompt: str, *, timeout: int = 60) -> str:
        httpx = _httpx()
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model":  self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0},  # deterministic output
                },
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            if "response" not in data:
                raise ConnectionError(
                    f"Unexpected response format from Ollama (missing 'response' key): {data}"
                )
            return data["response"]
        except httpx.ConnectError:
            raise ConnectionError(
                f"Could not connect to Ollama service: {self.base_url}\n"
                "Is the service running? Check: ollama serve"
            )

    def list_models(self) -> list[str]:
        httpx = _httpx()
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except httpx.ConnectError:
            raise ConnectionError(f"Could not connect to Ollama service: {self.base_url}")

    def pull_model(
        self,
        model: str,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        """
        Download a model via Ollama.
        Progress status is reported via the streaming response.
        """
        httpx = _httpx()
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/api/pull",
                json={"name": model},
                timeout=None,   # download duration is unpredictable
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if on_progress:
                        on_progress(PullProgress(
                            status=data.get("status", ""),
                            completed=data.get("completed", 0),
                            total=data.get("total", 0),
                        ))
        except httpx.ConnectError:
            raise ConnectionError(f"Could not connect to Ollama service: {self.base_url}")

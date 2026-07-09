from __future__ import annotations

import json
import logging
from typing import Any

from wardcat.llm.backends.base import BaseLLMBackend, ProgressCallback, PullProgress

logger = logging.getLogger(__name__)

_LOCALHOST_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


def _warn_if_http(url: str, allow_http: bool = False) -> None:
    """Enforce HTTPS for remote hosts; allow loopback HTTP silently.

    Loopback HTTP (``localhost``, ``127.0.0.1``, ``::1``) never leaves the
    machine, so there is no plaintext-on-the-wire risk — it is allowed with no
    warning and no ``allow_http`` needed (the common local-Ollama case). A remote
    HTTP connection would send PII across the network in the clear, so it raises
    ``ValueError`` unless ``allow_http=True`` is passed to opt in explicitly.
    """
    if not url.startswith("http://"):
        return
    if allow_http:
        return
    host = url[len("http://") :].split("/")[0].split(":")[0]
    if host in _LOCALHOST_HOSTS:
        return
    raise ValueError(
        f"LLM backend HTTP connection to remote host is not allowed: {url}\n"
        "PII would be transmitted unencrypted over the network.\n"
        "Use HTTPS, or pass allow_http=True to with_llm(...) to override (not recommended)."
    )


def _httpx() -> Any:
    try:
        import httpx

        return httpx
    except ImportError:
        raise ImportError(
            "'httpx' (a core wardcat dependency) is not importable — reinstall wardcat."
        ) from None


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
        allow_http: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        _warn_if_http(self.base_url, allow_http)

    def _friendly_http_error(self, exc: Exception) -> Exception:
        """Turn a raw HTTP error into an actionable one for the common cases.

        A request for a model Ollama hasn't pulled comes back as 404 / "model not
        found"; surface the installed models and the exact pull command instead of
        a bare status error.
        """
        response = getattr(exc, "response", None)
        if response is None:
            return exc
        body = ""
        try:
            body = response.json().get("error", "")
        except Exception:
            body = getattr(response, "text", "") or ""
        if response.status_code == 404 or "not found" in body.lower():
            try:
                available = self.list_models()
            except Exception:
                available = []
            hint = f" Installed models: {', '.join(available)}." if available else ""
            return ConnectionError(
                f"Ollama has no model named {self.model!r}.{hint} "
                f"Pull it with: ollama pull {self.model}"
            )
        return exc

    def complete(self, prompt: str, *, timeout: int = 60) -> str:
        httpx = _httpx()
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
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
            ) from None
        except httpx.HTTPStatusError as exc:
            raise self._friendly_http_error(exc) from None

    async def complete_async(self, prompt: str, *, timeout: int = 60) -> str:
        """Native async variant using ``httpx.AsyncClient``."""
        httpx = _httpx()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0},
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
            ) from None
        except httpx.HTTPStatusError as exc:
            raise self._friendly_http_error(exc) from None

    def list_models(self) -> list[str]:
        httpx = _httpx()
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except httpx.ConnectError:
            raise ConnectionError(f"Could not connect to Ollama service: {self.base_url}") from None

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
                timeout=None,  # download duration is unpredictable
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if on_progress:
                        on_progress(
                            PullProgress(
                                status=data.get("status", ""),
                                completed=data.get("completed", 0),
                                total=data.get("total", 0),
                            )
                        )
        except httpx.ConnectError:
            raise ConnectionError(f"Could not connect to Ollama service: {self.base_url}") from None

"""
vLLM backend — talks to a vLLM server over its OpenAI-compatible API.

vLLM (https://github.com/vllm-project/vllm) is a high-throughput on-prem serving
engine that exposes an OpenAI-compatible HTTP API. This backend reuses
:class:`~wardcat.llm.backends.openai_compat.OpenAICompatBackend` for transport
and adds vLLM-appropriate defaults plus a native chat path.

Start a server, then point wardcat at it::

    vllm serve meta-llama/Llama-3.1-8B-Instruct        # default port 8000

    from wardcat import Wardcat, Backend

    guard = Wardcat(salt="s").with_llm(
        backend=Backend.VLLM,
        model="meta-llama/Llama-3.1-8B-Instruct",
        base_url="http://localhost:8000/v1",
    )

The model must already be served by vLLM — model management is server-side, so
:meth:`pull_model` raises (inherited).
"""

from __future__ import annotations

import logging
from typing import Any

from wardcat.llm.backends.openai_compat import OpenAICompatBackend, _httpx

logger = logging.getLogger(__name__)

#: vLLM's default OpenAI-compatible endpoint (port 8000 + the ``/v1`` prefix).
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"


def _extract_content(data: dict[str, Any]) -> str:
    """Pull the assistant message out of an OpenAI-style chat response."""
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ConnectionError(
            f"Unexpected response format from vLLM service (missing expected keys): {exc}"
        ) from exc


class VLLMBackend(OpenAICompatBackend):
    """vLLM server backend (OpenAI-compatible API).

    Differs from the generic :class:`OpenAICompatBackend` in two ways:

    * **vLLM defaults** — ``base_url`` defaults to ``http://localhost:8000/v1``
      (vLLM's default port + ``/v1``); ``api_key`` is optional, since vLLM runs
      open unless started with ``--api-key``.
    * **Native chat** — :meth:`complete_messages` posts the real ``messages``
      array (system/user roles preserved) to ``/chat/completions`` instead of
      the base class's flattened single-prompt fallback. This matters for
      instruct models served with a chat template.

    The served model name must be passed as ``model`` (there is no universal
    default — it depends on what the vLLM server was started with).
    """

    def __init__(
        self,
        base_url: str = DEFAULT_VLLM_BASE_URL,
        model: str = "",
        api_key: str = "",
        allow_http: bool = False,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            allow_http=allow_http,
        )

    def complete_messages(self, messages: list[dict], *, timeout: int = 60) -> str:
        """Send the chat messages array natively to ``/chat/completions``."""
        httpx = _httpx()
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers,
                json={"model": self.model, "messages": messages, "temperature": 0},
                timeout=timeout,
            )
            response.raise_for_status()
            return _extract_content(response.json())
        except httpx.ConnectError:
            raise ConnectionError(f"Could not connect to vLLM service: {self.base_url}") from None

    async def complete_messages_async(self, messages: list[dict], *, timeout: int = 60) -> str:
        """Native async variant using ``httpx.AsyncClient``."""
        httpx = _httpx()
        try:
            async with httpx.AsyncClient(headers=self._headers) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json={"model": self.model, "messages": messages, "temperature": 0},
                    timeout=timeout,
                )
                response.raise_for_status()
                return _extract_content(response.json())
        except httpx.ConnectError:
            raise ConnectionError(f"Could not connect to vLLM service: {self.base_url}") from None

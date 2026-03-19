from __future__ import annotations

from ai_guard.llm.backends.base import BaseLLMBackend, ProgressCallback


def _httpx():
    try:
        import httpx
        return httpx
    except ImportError:
        raise ImportError(
            "LLM dedektörü için 'httpx' gerekli. "
            "Kurmak için: uv add 'ai-guard[llm]'"
        )


class OpenAICompatBackend(BaseLLMBackend):
    """
    OpenAI uyumlu REST API backend'i.

    vLLM, LM Studio, LocalAI, LiteLLM gibi servisleri destekler.
    Bu backend'lerde model indirme API aracılığıyla yapılamaz;
    sunucu tarafında yönetilmesi gerekir.
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
                f"LLM servisine bağlanılamadı: {self.base_url}"
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
            raise ConnectionError(f"LLM servisine bağlanılamadı: {self.base_url}")

    def pull_model(
        self,
        model: str,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        raise NotImplementedError(
            "OpenAI-uyumlu backend'de model indirme desteklenmez. "
            "Modeli sunucu tarafında (vLLM, LM Studio vb.) yönetin."
        )

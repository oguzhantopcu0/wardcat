from __future__ import annotations

import json

from ai_guard.llm.backends.base import BaseLLMBackend, ProgressCallback, PullProgress


def _httpx():
    try:
        import httpx
        return httpx
    except ImportError:
        raise ImportError(
            "LLM dedektörü için 'httpx' gerekli. "
            "Kurmak için: uv add 'ai-guard[llm]'"
        )


class OllamaBackend(BaseLLMBackend):
    """
    Ollama REST API backend'i.

    Varsayılan: http://localhost:11434
    Ollama, modeli yerel olarak çalıştırır ve indirme yönetimi sağlar.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def complete(self, prompt: str, *, timeout: int = 60) -> str:
        httpx = _httpx()
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model":  self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0},  # deterministik çıktı
                },
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()["response"]
        except httpx.ConnectError:
            raise ConnectionError(
                f"Ollama servisine bağlanılamadı: {self.base_url}\n"
                "Servis çalışıyor mu? Kontrol: ollama serve"
            )

    def list_models(self) -> list[str]:
        httpx = _httpx()
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except httpx.ConnectError:
            raise ConnectionError(f"Ollama servisine bağlanılamadı: {self.base_url}")

    def pull_model(
        self,
        model: str,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        """
        Ollama üzerinden model indir.
        Streaming yanıt ile ilerleme durumu raporlanır.
        """
        httpx = _httpx()
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/api/pull",
                json={"name": model},
                timeout=None,   # indirme süresi öngörülemez
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
            raise ConnectionError(f"Ollama servisine bağlanılamadı: {self.base_url}")

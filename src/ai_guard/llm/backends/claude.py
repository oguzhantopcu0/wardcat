"""
Claude API backend — Anthropic SDK üzerinden çalışır.

Gerçek bir Ollama/vLLM servisi olmadan LLMDetector mantığını test etmek için
kullanılır. ANTHROPIC_API_KEY ortam değişkeni gerektirir.

Kullanım:
    from ai_guard.llm.backends.claude import ClaudeBackend
    backend = ClaudeBackend()                        # claude-opus-4-6
    backend = ClaudeBackend(model="claude-haiku-4-5")  # daha hızlı/ucuz
"""
from __future__ import annotations

from .base import BaseLLMBackend, ProgressCallback


class ClaudeBackend(BaseLLMBackend):
    """
    Anthropic Claude API'yi BaseLLMBackend arayüzüyle sarar.

    ``complete()`` çağrısı system+user prompt'unu Claude'a gönderir ve
    metin yanıtını döndürür — tıpkı OllamaBackend gibi.

    pull_model() ve list_models() bu backend için anlamsızdır;
    NotImplementedError fırlatır (OllamaBackend ile simetrik).
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        max_tokens: int = 1024,
    ) -> None:
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "ClaudeBackend için 'anthropic' paketi gerekli: "
                "pip install anthropic"
            ) from exc

        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()  # ANTHROPIC_API_KEY env'den okunur

    # ------------------------------------------------------------------
    # BaseLLMBackend implementasyonu
    # ------------------------------------------------------------------

    def complete(self, prompt: str, *, timeout: int = 60) -> str:
        """
        Prompt'u Claude'a gönder ve metin yanıtını döndür.

        LLMDetector'ın build_prompt() çıktısı tek bir string'dir
        (system + user birleşik). Claude API'de system/user ayrımı
        zorunlu olduğundan, prompt'u ilk boş satırdan sonraki bloğa
        göre ayırıyoruz — ya da tamamını user mesajı olarak gönderiyoruz.
        """
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # İlk TextBlock'un metnini döndür
        for block in message.content:
            if block.type == "text":
                return block.text
        return ""

    def list_models(self) -> list[str]:
        """Claude API için model listesi döndürür (sabit)."""
        return ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"]

    def pull_model(
        self,
        model: str,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        """Claude API'de model indirme geçersizdir."""
        raise NotImplementedError(
            "ClaudeBackend için pull_model() desteklenmez. "
            "Modeller Anthropic API üzerinden hazır gelir."
        )

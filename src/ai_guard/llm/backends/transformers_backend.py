"""
HuggingFace Transformers backend — on-prem GPU/CPU inference.

Desteklenen modeller:
    meta-llama/Llama-3.2-1B-Instruct   — test / CPU (~2 GB RAM)
    meta-llama/Llama-3.2-3B-Instruct   — ~6 GB VRAM
    meta-llama/Llama-3.1-8B-Instruct   — ~16 GB VRAM  (8-bit: ~8 GB)
    meta-llama/Llama-3.1-70B-Instruct  — multi-GPU / 4-bit gerektirir

Kurulum::

    pip install "ai-guard[transformers]"

Kullanım::

    from ai_guard import LLMGuard

    guard = LLMGuard(
        use_llm=True,
        llm_backend="transformers",
        llm_model="meta-llama/Llama-3.1-8B-Instruct",
    )

8-bit / 4-bit quantization (VRAM tasarrufu)::

    guard = LLMGuard(
        use_llm=True,
        llm_backend="transformers",
        llm_model="meta-llama/Llama-3.1-8B-Instruct",
        llm_load_in_8bit=True,   # 16 GB → ~8 GB
    )

Ortam değişkenleri:
    HF_TOKEN   — gated modeller için HuggingFace erişim token'ı
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from ai_guard.llm.backends.base import BaseLLMBackend, ProgressCallback

logger = logging.getLogger(__name__)


class TransformersBackend(BaseLLMBackend):
    """
    HuggingFace Transformers tabanlı on-prem LLM backend.

    Model ilk ``complete_messages()`` çağrısında yüklenir (lazy loading).
    Thread-safe: model yükleme lock ile korunur.

    Args:
        model:         HuggingFace model ID (ör. ``"meta-llama/Llama-3.1-8B-Instruct"``).
        device_map:    ``"auto"`` çoklu GPU/CPU dağılımı için (varsayılan),
                       ``"cpu"`` sadece CPU için.
        load_in_8bit:  bitsandbytes 8-bit quantization — VRAM kullanımını ~yarıya indirir.
        load_in_4bit:  bitsandbytes 4-bit quantization — VRAM kullanımını ~çeyreğe indirir.
        max_new_tokens: Üretilecek maksimum token sayısı (varsayılan 512).
    """

    def __init__(
        self,
        model: str,
        *,
        device_map: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_new_tokens: int = 512,
    ) -> None:
        self._model_name    = model
        self._device_map    = device_map
        self._load_in_8bit  = load_in_8bit
        self._load_in_4bit  = load_in_4bit
        self._max_new_tokens = max_new_tokens
        self._pipeline: Any  = None
        self._lock           = threading.Lock()

    # ------------------------------------------------------------------
    # BaseLLMBackend interface
    # ------------------------------------------------------------------

    def complete(self, prompt: str, *, timeout: int = 60) -> str:
        """Plain-text prompt'u user mesajı olarak gönder."""
        return self.complete_messages(
            [{"role": "user", "content": prompt}],
            timeout=timeout,
        )

    def complete_messages(self, messages: list[dict], *, timeout: int = 60) -> str:
        """Chat mesajlarını tokenizer chat template ile modele gönder.

        Args:
            messages: ``[{"role": "system"|"user"|"assistant", "content": ...}, ...]``
            timeout:  Kullanılmaz (local inference'ta network timeout yoktur);
                      arayüz uyumluluğu için parametre korunur.

        Returns:
            Modelin ürettiği assistant yanıtı (ham metin).
        """
        pipe = self._get_pipeline()

        outputs = pipe(
            messages,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,          # temperature=0 eşdeğeri — deterministik
            return_full_text=False,   # sadece yeni token'lar döner
        )

        # Transformers pipeline çıktı formatı:
        # [{"generated_text": "..." | [{"role":..., "content":...}]}]
        generated = outputs[0]["generated_text"]
        if isinstance(generated, list):
            # Chat format: son mesajın içeriğini döndür
            return generated[-1].get("content", "")
        return str(generated)

    def list_models(self) -> list[str]:
        """Yüklü modelin adını döndür."""
        return [self._model_name]

    def pull_model(
        self,
        model: str,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        """Modeli HuggingFace Hub'dan indir.

        Args:
            model:       HuggingFace model ID (ör. ``"meta-llama/Llama-3.1-8B-Instruct"``).
            on_progress: İlerleme callback'i (HF indirme detaylı ilerleme sunmaz —
                         başlangıç ve bitiş olayları gönderilir).

        Raises:
            ImportError:   ``huggingface_hub`` kurulu değilse.
            OSError:       Gated model için HF_TOKEN eksikse.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "Model indirmek için 'huggingface_hub' gerekli.\n"
                "Kurmak için: pip install huggingface_hub"
            )

        logger.info("HuggingFace Hub'dan indiriliyor: %s", model)
        if on_progress:
            from ai_guard.llm.backends.base import PullProgress
            on_progress(PullProgress(status="downloading", completed=0, total=0))

        snapshot_download(repo_id=model)

        logger.info("Model indirildi: %s", model)
        if on_progress:
            from ai_guard.llm.backends.base import PullProgress
            on_progress(PullProgress(status="success", completed=1, total=1))

    # ------------------------------------------------------------------
    # Dahili
    # ------------------------------------------------------------------

    def _get_pipeline(self) -> Any:
        """Model pipeline'ını döndür; henüz yüklenmemişse yükle (lazy, thread-safe)."""
        with self._lock:
            if self._pipeline is None:
                self._pipeline = self._load_pipeline()
        return self._pipeline

    def _load_pipeline(self) -> Any:
        """transformers pipeline'ını oluştur ve döndür."""
        try:
            import torch
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "Transformers backend için 'transformers' ve 'torch' gerekli.\n"
                "Kurmak için: pip install 'ai-guard[transformers]'"
            )

        logger.info("Transformers modeli yükleniyor: %s", self._model_name)

        kwargs: dict[str, Any] = {
            "task":        "text-generation",
            "model":       self._model_name,
            "device_map":  self._device_map,
            "torch_dtype": torch.bfloat16,
        }

        if self._load_in_8bit:
            kwargs["load_in_8bit"] = True
        elif self._load_in_4bit:
            kwargs["load_in_4bit"] = True

        pipe = pipeline(**kwargs)
        logger.info("Transformers modeli hazır: %s", self._model_name)
        return pipe

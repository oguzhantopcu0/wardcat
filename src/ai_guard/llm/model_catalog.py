"""
Önerilen on-prem LLM modelleri kataloğu.

Kullanıcıya ``models setup`` komutu veya ``LLMGuard(use_llm=True)`` ile
sunulan seçenekler burada tanımlanır.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """Katalogdaki bir on-prem LLM modelinin meta verisi."""

    name:        str
    """Model adı — Ollama için ``"llama3.1:8b"``, Transformers için HF model ID."""
    vram_gb:     float
    """Yaklaşık VRAM gereksinimi GB cinsinden."""
    description: str
    """Kısa kullanıcıya yönelik açıklama."""
    backend:     str = "ollama"
    """Backend adı: ``"ollama"`` veya ``"transformers"``."""
    recommended: bool = False
    """``True`` ise ``models setup`` varsayılan olarak bu modeli önerir."""


# Desteklenen modeller — backend ve VRAM'a göre gruplandırılmış
CATALOG: list[ModelInfo] = [
    # ── Ollama modelleri (GGUF, Q4_K_M) ─────────────────────────────
    ModelInfo(
        name        = "llama3.1:8b",
        vram_gb     = 4.7,
        description = "Llama 3.1 8B · JSON kalitesi yüksek · GTX 1070+ önerilir",
        backend     = "ollama",
        recommended = True,
    ),
    ModelInfo(
        name        = "llama3.2:3b",
        vram_gb     = 2.0,
        description = "Llama 3.2 3B · Hızlı ve hafif · 4 GB VRAM'da çalışır",
        backend     = "ollama",
    ),
    ModelInfo(
        name        = "mistral:7b",
        vram_gb     = 4.1,
        description = "Mistral 7B · İyi JSON uyumu · Alternatif 7B seçenek",
        backend     = "ollama",
    ),
    ModelInfo(
        name        = "phi3:mini",
        vram_gb     = 2.5,
        description = "Phi-3 Mini 3.8B · Çok hızlı · Düşük VRAM gereksinimi",
        backend     = "ollama",
    ),

    # ── HuggingFace Transformers (Llama, tam hassasiyet) ─────────────
    ModelInfo(
        name        = "meta-llama/Llama-3.2-1B-Instruct",
        vram_gb     = 2.5,
        description = "Llama 3.2 1B Instruct · Test / CPU · pip install ai-guard[transformers]",
        backend     = "transformers",
    ),
    ModelInfo(
        name        = "meta-llama/Llama-3.2-3B-Instruct",
        vram_gb     = 6.0,
        description = "Llama 3.2 3B Instruct · ~6 GB VRAM · pip install ai-guard[transformers]",
        backend     = "transformers",
    ),
    ModelInfo(
        name        = "meta-llama/Llama-3.1-8B-Instruct",
        vram_gb     = 16.0,
        description = "Llama 3.1 8B Instruct · 16 GB VRAM (8-bit: ~8 GB) · pip install ai-guard[transformers]",
        backend     = "transformers",
    ),
    ModelInfo(
        name        = "meta-llama/Llama-3.1-70B-Instruct",
        vram_gb     = 40.0,
        description = "Llama 3.1 70B Instruct · Multi-GPU / 4-bit gerektirir · pip install ai-guard[transformers]",
        backend     = "transformers",
    ),
]

DEFAULT_MODEL = "llama3.1:8b"


def get_model(name: str) -> ModelInfo | None:
    """Katalogdan model bilgisi döndür; bulunamazsa None."""
    for m in CATALOG:
        if m.name == name:
            return m
    return None


def recommended() -> ModelInfo:
    """Varsayılan önerilen modeli döndür."""
    for m in CATALOG:
        if m.recommended:
            return m
    return CATALOG[0]

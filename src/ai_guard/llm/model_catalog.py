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
    """Ollama model adı, örn. ``"llama3.1:8b"``."""
    vram_gb:     float
    """Yaklaşık VRAM gereksinimi GB cinsinden (Q4_K_M kuantizasyonu)."""
    description: str
    """Kısa kullanıcıya yönelik açıklama."""
    recommended: bool = False
    """``True`` ise ``models setup`` varsayılan olarak bu modeli önerir."""


# Desteklenen modeller — VRAM'a göre sıralı
CATALOG: list[ModelInfo] = [
    ModelInfo(
        name        = "llama3.1:8b",
        vram_gb     = 4.7,
        description = "Llama 3.1 8B · JSON kalitesi yüksek · GTX 1070+ önerilir",
        recommended = True,
    ),
    ModelInfo(
        name        = "llama3.2:3b",
        vram_gb     = 2.0,
        description = "Llama 3.2 3B · Hızlı ve hafif · 4 GB VRAM'da çalışır",
    ),
    ModelInfo(
        name        = "mistral:7b",
        vram_gb     = 4.1,
        description = "Mistral 7B · İyi JSON uyumu · Alternatif 7B seçenek",
    ),
    ModelInfo(
        name        = "phi3:mini",
        vram_gb     = 2.5,
        description = "Phi-3 Mini 3.8B · Çok hızlı · Düşük VRAM gereksinimi",
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

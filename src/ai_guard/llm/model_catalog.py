"""
Catalog of recommended on-prem LLM models.

The options presented to the user via the ``models setup`` command or
``LLMGuard(use_llm=True)`` are defined here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a single on-prem LLM model in the catalog."""

    name: str
    """Model name — e.g. ``"llama3.1:8b"`` for Ollama, HF model ID for Transformers."""
    vram_gb: float
    """Approximate VRAM requirement in GB."""
    description: str
    """Short user-facing description."""
    backend: str = "ollama"
    """Backend name: ``"ollama"`` or ``"transformers"``."""
    recommended: bool = False
    """``True`` if ``models setup`` recommends this model by default."""


# Supported models — grouped by backend and VRAM
CATALOG: list[ModelInfo] = [
    # ── Ollama models (GGUF, Q4_K_M) ─────────────────────────────────
    ModelInfo(
        name="llama3.1:8b",
        vram_gb=4.7,
        description="Llama 3.1 8B · High JSON quality · GTX 1070+ recommended",
        backend="ollama",
        recommended=True,
    ),
    ModelInfo(
        name="llama3.2:3b",
        vram_gb=2.0,
        description="Llama 3.2 3B · Fast and lightweight · Runs on 4 GB VRAM",
        backend="ollama",
    ),
    ModelInfo(
        name="mistral:7b",
        vram_gb=4.1,
        description="Mistral 7B · Good JSON compliance · Alternative 7B option",
        backend="ollama",
    ),
    ModelInfo(
        name="phi3:mini",
        vram_gb=2.5,
        description="Phi-3 Mini 3.8B · Very fast · Low VRAM requirement",
        backend="ollama",
    ),
    # ── HuggingFace Transformers (Llama, full precision) ──────────────
    ModelInfo(
        name="meta-llama/Llama-3.2-1B-Instruct",
        vram_gb=2.5,
        description="Llama 3.2 1B Instruct · Test / CPU · pip install ai-guard[transformers]",
        backend="transformers",
    ),
    ModelInfo(
        name="meta-llama/Llama-3.2-3B-Instruct",
        vram_gb=6.0,
        description="Llama 3.2 3B Instruct · ~6 GB VRAM · pip install ai-guard[transformers]",
        backend="transformers",
    ),
    ModelInfo(
        name="meta-llama/Llama-3.1-8B-Instruct",
        vram_gb=16.0,
        description="Llama 3.1 8B Instruct · 16 GB VRAM (8-bit: ~8 GB) · pip install ai-guard[transformers]",
        backend="transformers",
    ),
    ModelInfo(
        name="meta-llama/Llama-3.1-70B-Instruct",
        vram_gb=40.0,
        description="Llama 3.1 70B Instruct · Multi-GPU / requires 4-bit · pip install ai-guard[transformers]",
        backend="transformers",
    ),
]

DEFAULT_MODEL = "llama3.1:8b"


def get_model(name: str) -> ModelInfo | None:
    """Return model info from catalog; None if not found."""
    for m in CATALOG:
        if m.name == name:
            return m
    return None


def recommended() -> ModelInfo:
    """Return the default recommended model."""
    for m in CATALOG:
        if m.recommended:
            return m
    return CATALOG[0]

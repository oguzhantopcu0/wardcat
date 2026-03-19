"""
Model yönetimi: indirme, listeleme, kullanılabilirlik kontrolü.
"""
from __future__ import annotations

import sys

from ai_guard.llm.backends.base import BaseLLMBackend, PullProgress


class ModelManager:
    """
    On-prem Llama model yöneticisi.

    Örnek::

        from ai_guard.llm.backends.ollama import OllamaBackend
        from ai_guard.llm.model_manager import ModelManager

        mgr = ModelManager(OllamaBackend())
        mgr.pull("llama3.2")
        print(mgr.list())
    """

    def __init__(self, backend: BaseLLMBackend) -> None:
        self.backend = backend

    def list(self) -> list[str]:
        """Serviste mevcut modelleri listele."""
        return self.backend.list_models()

    def is_available(self, model: str) -> bool:
        """Model serviste kullanıma hazır mı?"""
        return self.backend.is_model_available(model)

    def ensure_available(self, model: str, *, verbose: bool = True) -> bool:
        """
        Model mevcut değilse indir.

        :param model:   Model adı, örn. "llama3.1:8b"
        :param verbose: İlerlemeyi ve bildirimleri terminale yaz
        :returns:       True — model hazır; False — indirme iptal edildi veya hata
        :raises NotImplementedError: Backend model indirmeyi desteklemiyorsa
        """
        if self.is_available(model):
            if verbose:
                print(f"  ✓ Model zaten mevcut: {model}")
            return True

        if verbose:
            print(f"  Model bulunamadı: {model}")
            try:
                answer = input(f"  İndirilsin mi? (~GB düzeyinde) [e/H]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = ""
            if answer not in ("e", "evet", "y", "yes"):
                print("  İndirme iptal edildi.")
                return False

        self.pull(model, verbose=verbose)
        return True

    def pull(self, model: str, *, verbose: bool = True) -> None:
        """
        Modeli indir.

        :param model:   Model adı, örn. "llama3.2" veya "llama3.2:3b"
        :param verbose: İlerlemeyi terminale yaz
        """
        if verbose:
            print(f"Model indiriliyor: {model}")

        def _on_progress(p: PullProgress) -> None:
            if not verbose:
                return
            if p.total:
                bar_len  = 30
                filled   = int(bar_len * p.completed / p.total)
                bar      = "█" * filled + "░" * (bar_len - filled)
                print(f"\r  [{bar}] {p.percent:5.1f}%  {p.status:<20}", end="", flush=True)
            else:
                print(f"\r  {p.status:<40}", end="", flush=True)

        self.backend.pull_model(model, on_progress=_on_progress)

        if verbose:
            print(f"\r  Model hazır: {model}{' ' * 40}")

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Iterator


@dataclass
class PullProgress:
    """Model indirme ilerlemesi."""
    status:    str
    completed: int = 0
    total:     int = 0

    @property
    def percent(self) -> float:
        return (self.completed / self.total * 100) if self.total else 0.0


ProgressCallback = Callable[[PullProgress], None]


class BaseLLMBackend(ABC):
    """Tüm LLM backend'lerinin uyması gereken arayüz."""

    @abstractmethod
    def complete(self, prompt: str, *, timeout: int = 60) -> str:
        """Prompt gönder, tamamlama metnini döndür."""
        ...

    @abstractmethod
    def list_models(self) -> list[str]:
        """Kullanılabilir model isimlerini listele."""
        ...

    @abstractmethod
    def pull_model(
        self,
        model: str,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        """Modeli indir. İlerleme için on_progress callback'i çağrılır."""
        ...

    def is_model_available(self, model: str) -> bool:
        """Modelin serviste hazır olup olmadığını döndür."""
        return model in self.list_models()

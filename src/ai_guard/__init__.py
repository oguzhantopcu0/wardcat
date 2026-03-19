import logging
from importlib.metadata import PackageNotFoundError, version

from ai_guard.core.models import Action, ScanResult, Violation
from ai_guard.guard import LLMGuard

try:
    __version__: str = version("ai-guard")
except PackageNotFoundError:
    __version__ = "0.2.0"  # geliştirme ortamı fallback

__all__ = ["LLMGuard", "ScanResult", "Violation", "Action", "__version__"]

# Kütüphane logging best-practice: NullHandler eklenir, handler'ı uygulama konfigüre eder.
logging.getLogger(__name__).addHandler(logging.NullHandler())

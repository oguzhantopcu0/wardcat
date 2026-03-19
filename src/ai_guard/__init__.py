import logging

from ai_guard.core.models import Action, ScanResult, Violation
from ai_guard.guard import LLMGuard

__version__ = "0.1.0"
__all__ = ["LLMGuard", "ScanResult", "Violation", "Action", "__version__"]

# Kütüphane logging best-practice: NullHandler eklenir, handler'ı uygulama konfigüre eder.
logging.getLogger(__name__).addHandler(logging.NullHandler())

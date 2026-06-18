import logging
from importlib.metadata import PackageNotFoundError, version

from ai_guard.core.models import Action, ScanResult, Violation
from ai_guard.entity_groups import (
    all_entities,
    core_entities,
    european_entities,
    financial_entities,
    identity_entities,
    network_entities,
    turkish_entities,
    uk_entities,
    us_entities,
)
from ai_guard.guard import LLMGuard

try:
    __version__: str = version("ai-guard")
except PackageNotFoundError:
    __version__ = "0.2.0b1"  # development environment fallback

__all__ = [
    "LLMGuard",
    "ScanResult",
    "Violation",
    "Action",
    "__version__",
    "redacted",
    # Entity group helpers
    "core_entities",
    "financial_entities",
    "turkish_entities",
    "european_entities",
    "uk_entities",
    "us_entities",
    "network_entities",
    "identity_entities",
    "all_entities",
]


def redacted(result: "ScanResult") -> dict:  # noqa: F821
    """Convenience wrapper — equivalent to ``result.redacted()``."""
    return result.redacted()


# Library logging best practice: NullHandler is added; the application configures the handler.
logging.getLogger(__name__).addHandler(logging.NullHandler())

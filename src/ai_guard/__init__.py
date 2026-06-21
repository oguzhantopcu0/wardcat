import logging
from importlib.metadata import PackageNotFoundError, version

from ai_guard.core.models import KNOWN_ENTITY_TYPES, Action, Entity, ScanResult, Violation
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
from ai_guard.exceptions import (
    AIGuardError,
    ConfigError,
    ModelDownloadError,
    UnsupportedLanguageError,
)
from ai_guard.guard import AIGuard
from ai_guard.ner.spacy_catalog import Language

try:
    __version__: str = version("ai-guard")
except PackageNotFoundError:
    __version__ = "0.4.0"  # development environment fallback

__all__ = [
    "AIGuard",
    "ScanResult",
    "Violation",
    "Action",
    "Entity",
    "Language",
    "KNOWN_ENTITY_TYPES",
    "__version__",
    "redacted",
    # Exceptions
    "AIGuardError",
    "ConfigError",
    "ModelDownloadError",
    "UnsupportedLanguageError",
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

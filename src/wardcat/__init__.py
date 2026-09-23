import logging
from importlib.metadata import PackageNotFoundError, version

from wardcat.core.actions import (
    ActionContext,
    TokenAllocator,
    register_action,
    registered_actions,
)
from wardcat.core.models import (
    KNOWN_ENTITY_TYPES,
    SENSITIVITY_CATEGORIES,
    Action,
    Entity,
    Layer,
    RedactedResult,
    RedactedViolation,
    ScanResult,
    SensitivityVerdict,
    Violation,
)
from wardcat.core.restore import RestoredText, Substitution, UnrestoredValue
from wardcat.entity_groups import (
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
from wardcat.exceptions import (
    ConfigError,
    ContextMismatch,
    DegradedScanError,
    ModelDownloadError,
    UnsupportedLanguageError,
    WardcatError,
)
from wardcat.guard import Wardcat
from wardcat.llm.backends.base import Backend
from wardcat.llm.circuit import CircuitOpen
from wardcat.ner.spacy_catalog import Language, supported_languages
from wardcat.presets import Preset, supported_presets

try:
    __version__: str = version("wardcat")
except PackageNotFoundError:
    __version__ = "1.2.1"  # development environment fallback

__all__ = [
    "Wardcat",
    "ScanResult",
    "Violation",
    "RestoredText",
    "Substitution",
    "UnrestoredValue",
    "RedactedResult",
    "RedactedViolation",
    "Action",
    "Entity",
    "Layer",
    "SensitivityVerdict",
    "SENSITIVITY_CATEGORIES",
    "Language",
    "supported_languages",
    "Preset",
    "supported_presets",
    "Backend",
    "register_action",
    "registered_actions",
    "ActionContext",
    "TokenAllocator",
    "KNOWN_ENTITY_TYPES",
    "__version__",
    "redacted",
    # Exceptions
    "WardcatError",
    "ConfigError",
    "ContextMismatch",
    "DegradedScanError",
    "CircuitOpen",
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


def redacted(result: "ScanResult") -> "RedactedResult":  # noqa: F821
    """Convenience wrapper — equivalent to ``result.redacted()``."""
    return result.redacted()


# Library logging best practice: NullHandler is added; the application configures the handler.
logging.getLogger(__name__).addHandler(logging.NullHandler())

"""
Exception hierarchy for ai-guard.

All errors raised by the library derive from :class:`AIGuardError`, so callers
can catch everything ai-guard raises with a single ``except AIGuardError``.

For backward compatibility the concrete errors also subclass the built-in
exception they replaced (``ConfigError`` is a ``ValueError``,
``ModelDownloadError`` is a ``RuntimeError``), so existing ``except ValueError``
/ ``except RuntimeError`` code keeps working.
"""

from __future__ import annotations


class AIGuardError(Exception):
    """Base class for every error raised by ai-guard."""


class ConfigError(AIGuardError, ValueError):
    """Invalid configuration (bad action, backend, pattern, entity spec, …).

    Subclasses :class:`ValueError` for backward compatibility.
    """


class ModelDownloadError(AIGuardError, RuntimeError):
    """A SpaCy/LLM model could not be downloaded or is incompatible.

    Subclasses :class:`RuntimeError` for backward compatibility.
    """


class UnsupportedLanguageError(ConfigError):
    """The requested NER language (or size tier) has no compatible model."""

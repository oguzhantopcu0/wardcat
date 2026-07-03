"""
Exception hierarchy for wardcat.

All errors raised by the library derive from :class:`WardcatError`, so callers
can catch everything wardcat raises with a single ``except WardcatError``.

For backward compatibility the concrete errors also subclass the built-in
exception they replaced (``ConfigError`` is a ``ValueError``,
``ModelDownloadError`` is a ``RuntimeError``), so existing ``except ValueError``
/ ``except RuntimeError`` code keeps working.
"""

from __future__ import annotations


class WardcatError(Exception):
    """Base class for every error raised by wardcat."""


class ConfigError(WardcatError, ValueError):
    """Invalid configuration (bad action, backend, pattern, entity spec, …).

    Subclasses :class:`ValueError` for backward compatibility.
    """


class ModelDownloadError(WardcatError, RuntimeError):
    """A SpaCy/LLM model could not be downloaded or is incompatible.

    Subclasses :class:`RuntimeError` for backward compatibility.
    """


class UnsupportedLanguageError(ConfigError):
    """The requested NER language (or size tier) has no compatible model."""

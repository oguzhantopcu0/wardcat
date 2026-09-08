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


class ContextMismatch(WardcatError):
    """A text carries a reversible placeholder that belongs to a different scan.

    Raised by ``ScanResult.restore(..., strict=True)``. Every scan stamps its own
    context id into the placeholders it produces, so a token from another scan can
    be recognized rather than silently matched — restoring one request's answer
    against another request's result would otherwise substitute the wrong
    person's values. Nothing is substituted for the foreign tokens either way;
    ``strict`` only decides whether that is an error or a report.

    ``placeholders`` lists the foreign tokens found.
    """

    def __init__(self, placeholders: list[str]) -> None:
        self.placeholders = placeholders
        super().__init__(
            f"{len(placeholders)} placeholder(s) in this text belong to a different "
            f"scan and were left in place: {', '.join(sorted(placeholders))}. Restore "
            "with the ScanResult that produced them, or pass them in `also=[...]`."
        )


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

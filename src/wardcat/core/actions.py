"""Registry of anonymization actions — the extension point for new actions.

An action turns a detected span into its replacement (or ``None`` to keep the
text and only report it). Built-in actions are ``warn`` / ``hash`` / ``redact`` /
``mask`` / ``tokenize``; register your own (``encrypt``, format-preserving, …)
without touching the core::

    from wardcat import register_action

    register_action("vault", lambda span, ctx: f"<{span.entity_type}:{vault.put(span.text)}>")
    guard.add_entity("EMAIL", "vault")
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field

from wardcat.detectors.base import DetectedSpan
from wardcat.exceptions import ConfigError
from wardcat.utils.hashing import sha256_hash

#: Hex characters in a context id. Twelve is 48 bits: with a hundred thousand
#: results alive at once the chance that any two share an id is about 1.8e-5,
#: and request-scoped use (a few hundred alive) puts it near 1e-9.
_CONTEXT_ID_CHARS = 12


def new_context_id() -> str:
    """A fresh context id — the per-scan half of every reversible placeholder."""
    return secrets.token_hex(_CONTEXT_ID_CHARS // 2)


class TokenAllocator:
    """Hands out stable, unique placeholders for one scan — the ``tokenize`` vault.

    A placeholder is ``[TYPE_index_contextid]`` — ``[EMAIL_1_9f3a2c8b71d4]``. The
    index is per entity type in order of first appearance and restarts at 1 for
    every scan, so it stays short and readable; the context id is drawn once per
    allocator and shared by every token it hands out, which is what makes one
    scan's placeholders distinct from another's. An *identical* value always gets
    the same token, so a name repeated three times stays one referent for whatever
    reads the anonymized text.

    That distinctness is the safety property behind
    :meth:`~wardcat.ScanResult.restore`: two scans running side by side both hold
    an "``EMAIL`` number 1", and without the context id their placeholders would be
    the same string — restoring one request's answer against another's result
    would silently substitute the wrong person's value. With it there is nothing to
    match, so the mistake becomes a reported non-substitution instead.

    Allocation state is per instance and the
    :class:`Anonymizer <wardcat.core.anonymizer.Anonymizer>` builds a fresh one for
    every ``apply()`` call, so concurrent scans never share a counter or an id.

    :param context_id: the id to stamp into every token. Defaults to a fresh one;
        pass ``""`` for bare ``[TYPE_index]`` placeholders (no cross-scan
        protection), or a fixed value to make output reproducible in tests.

    .. warning::
        Holds the raw values it has seen for the lifetime of the instance.
    """

    def __init__(self, context_id: str | None = None) -> None:
        self.context_id = new_context_id() if context_id is None else context_id
        self._tokens: dict[tuple[str, str], str] = {}
        self._counts: dict[str, int] = {}

    def token_for(self, entity_type: str, text: str) -> str:
        """Return the placeholder for *text*, allocating a new one on first sight."""
        key = (entity_type, text)
        token = self._tokens.get(key)
        if token is None:
            count = self._counts.get(entity_type, 0) + 1
            self._counts[entity_type] = count
            suffix = f"_{self.context_id}" if self.context_id else ""
            token = f"[{entity_type}_{count}{suffix}]"
            self._tokens[key] = token
        return token


@dataclass(frozen=True)
class ActionContext:
    """Extra context an action may need beyond the span (e.g. the hash salt)."""

    salt: str = ""
    tokens: TokenAllocator = field(default_factory=TokenAllocator, repr=False, compare=False)
    """Placeholder vault for reversible actions, scoped to a single scan — its
    ``context_id`` is the one stamped into this scan's tokens. Excluded from
    ``repr``/equality so two contexts with the same salt still compare equal."""


#: An action maps ``(span, context)`` to a replacement string, or ``None`` to
#: keep the original text (report-only, like ``warn``).
ActionFn = Callable[[DetectedSpan, ActionContext], "str | None"]

_ACTIONS: dict[str, ActionFn] = {}


def register_action(name: str, fn: ActionFn) -> None:
    """Register (or override) an anonymization action under *name*."""
    _ACTIONS[name] = fn


def registered_actions() -> frozenset[str]:
    """The names of all currently-registered actions (built-in + custom)."""
    return frozenset(_ACTIONS)


def get_action(name: str) -> ActionFn:
    """Return the action function registered under *name*."""
    fn = _ACTIONS.get(name)
    if fn is None:
        raise ConfigError(
            f"Unknown action {name!r}. Registered actions: {sorted(_ACTIONS)}. "
            "Add one with wardcat.register_action(name, fn)."
        )
    return fn


# ── Built-in actions ──────────────────────────────────────────────────────────


def _act_warn(span: DetectedSpan, ctx: ActionContext) -> str | None:
    return None  # keep the text; report only


def _act_hash(span: DetectedSpan, ctx: ActionContext) -> str | None:
    digest = sha256_hash(span.text, ctx.salt)[:16]
    return f"[{span.entity_type}:{digest}]"


def _act_redact(span: DetectedSpan, ctx: ActionContext) -> str | None:
    return f"[{span.entity_type}]"


def _act_mask(span: DetectedSpan, ctx: ActionContext) -> str | None:
    return _mask_value(span.entity_type, span.text)


def _act_tokenize(span: DetectedSpan, ctx: ActionContext) -> str | None:
    return ctx.tokens.token_for(span.entity_type, span.text)


register_action("warn", _act_warn)
register_action("hash", _act_hash)
register_action("redact", _act_redact)
register_action("mask", _act_mask)
register_action("tokenize", _act_tokenize)


def _mask_value(entity_type: str, text: str) -> str:
    """Produce an entity-aware masked version of *text*.

    Masking rules per entity type:

    ============== =================================================
    CREDIT_CARD    Last 4 digits visible: ``************1111``
    EMAIL          First char + stars + full domain: ``u***@example.com``
    PHONE          Last 4 digits visible: ``*******5678``
    SSN            Standard US format: ``***-**-6789``
    IBAN           Country code + last 4: ``TR**...**1326``
    TC_ID          Last 3 digits: ``********950``
    NIN            Last 3: ``AB123***``
    *default*      First 2 + stars + last 2: ``ab****cd``
    ============== =================================================
    """
    n = len(text)
    if entity_type == "CREDIT_CARD":
        digits = re.sub(r"[^0-9]", "", text)
        if len(digits) >= 4:
            return "*" * (len(digits) - 4) + digits[-4:]
        return "*" * n

    if entity_type == "EMAIL":
        at = text.find("@")
        if at > 0:
            local = text[:at]
            domain = text[at:]
            masked_local = local[0] + "*" * max(len(local) - 1, 1)
            return masked_local + domain
        return "*" * n

    if entity_type == "PHONE":
        digits = re.sub(r"[^0-9]", "", text)
        if len(digits) >= 4:
            return "*" * (n - 4) + text[-4:]
        return "*" * n

    if entity_type == "SSN":
        digits = re.sub(r"[^0-9]", "", text)
        if len(digits) >= 4:
            return f"***-**-{digits[-4:]}"
        return "*" * n

    if entity_type == "IBAN":
        # Keep country code (2 chars) + last 4
        if n >= 6:
            return text[:2] + "*" * (n - 6) + text[-4:]
        return "*" * n

    if entity_type == "TC_ID":
        # Last 3 digits visible
        if n >= 3:
            return "*" * (n - 3) + text[-3:]
        return "*" * n

    if entity_type == "NIN":
        # Last 3 chars visible
        if n >= 3:
            return "*" * (n - 3) + text[-3:]
        return "*" * n

    # Default: first 2 + stars + last 2
    if n >= 4:
        return text[:2] + "*" * (n - 4) + text[-2:]
    return "*" * n

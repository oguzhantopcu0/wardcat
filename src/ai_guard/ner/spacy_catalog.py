"""
Catalog of supported SpaCy NER models.

Used by the ``ai-guard spacy list`` and ``ai-guard spacy download`` CLI commands.
Model names are the official SpaCy package names passed to ``python -m spacy download``.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpacyModelInfo:
    """Metadata for a single SpaCy model in the catalog."""

    name: str
    """Official SpaCy model package name, e.g. ``"en_core_web_sm"``."""
    language: str
    """Human-readable language name."""
    lang_code: str
    """ISO 639-1 language code prefix, e.g. ``"en"``, ``"tr"``."""
    size: str
    """Size tier: ``"sm"``, ``"md"``, ``"lg"``, or ``"trf"``."""
    ram_mb: int
    """Approximate RAM requirement in MB when loaded."""
    description: str
    """Short user-facing description."""
    recommended: bool = False
    """``True`` for the recommended default model of this language."""
    wheel_url: str = ""
    """Direct wheel URL (used when the model is not on PyPI/spacy.io, e.g. HuggingFace)."""
    spacy_compat: str = ""
    """SpaCy version constraint for this model, e.g. ``">=3.4,<3.5"``."""
    note: str = ""
    """Optional compatibility or installation note shown to the user."""
    extra_packages: tuple[str, ...] = ()
    """Additional pip packages to install after the wheel (e.g. ``("spacy-transformers",)``)."""


SPACY_CATALOG: list[SpacyModelInfo] = [
    # ── English ──────────────────────────────────────────────────────────
    SpacyModelInfo(
        name        = "en_core_web_sm",
        language    = "English",
        lang_code   = "en",
        size        = "sm",
        ram_mb      = 15,
        description = "Small · CNN pipeline · ~15 MB · Best for dev/testing",
        recommended = True,
    ),
    SpacyModelInfo(
        name        = "en_core_web_md",
        language    = "English",
        lang_code   = "en",
        size        = "md",
        ram_mb      = 50,
        description = "Medium · Word vectors included · ~50 MB · Better accuracy",
    ),
    SpacyModelInfo(
        name        = "en_core_web_lg",
        language    = "English",
        lang_code   = "en",
        size        = "lg",
        ram_mb      = 750,
        description = "Large · Full word vectors · ~750 MB · High accuracy",
    ),
    SpacyModelInfo(
        name        = "en_core_web_trf",
        language    = "English",
        lang_code   = "en",
        size        = "trf",
        ram_mb      = 440,
        description = "Transformer (RoBERTa) · Best accuracy · Requires GPU for speed",
    ),

    # ── Turkish ──────────────────────────────────────────────────────────
    # Hosted on HuggingFace by turkish-nlp-suite; requires SpaCy >=3.4,<3.5.
    # No sm model exists. No SpaCy 3.7/3.8 compatible release as of 2026-03.
    # Install with: uv pip install <wheel_url> --no-deps
    SpacyModelInfo(
        name        = "tr_core_news_md",
        language    = "Turkish",
        lang_code   = "tr",
        size        = "md",
        ram_mb      = 156,
        description = "Medium · Word vectors · ~156 MB · Recommended for Turkish PII",
        recommended = True,
        wheel_url   = "https://huggingface.co/turkish-nlp-suite/tr_core_news_md/resolve/main/tr_core_news_md-1.0-py3-none-any.whl",
        spacy_compat= ">=3.4,<3.5",
        note        = "Hosted on HuggingFace (v1.0). Built for SpaCy 3.4.x — installed with --no-deps on newer versions.",
    ),
    SpacyModelInfo(
        name        = "tr_core_news_lg",
        language    = "Turkish",
        lang_code   = "tr",
        size        = "lg",
        ram_mb      = 560,
        description = "Large · Full vectors · ~560 MB · Higher Turkish NER accuracy",
        wheel_url   = "https://huggingface.co/turkish-nlp-suite/tr_core_news_lg/resolve/main/tr_core_news_lg-1.0-py3-none-any.whl",
        spacy_compat= ">=3.4,<3.5",
        note        = "Hosted on HuggingFace (v1.0). Built for SpaCy 3.4.x — installed with --no-deps on newer versions.",
    ),
    SpacyModelInfo(
        name           = "tr_core_news_trf",
        language       = "Turkish",
        lang_code      = "tr",
        size           = "trf",
        ram_mb         = 850,
        description    = "Transformer (BERTurk) · Best Turkish accuracy · Requires GPU for speed",
        wheel_url      = "https://huggingface.co/turkish-nlp-suite/tr_core_news_trf/resolve/main/tr_core_news_trf-1.0-py3-none-any.whl",
        spacy_compat   = ">=3.4,<3.5",
        note           = "Hosted on HuggingFace (v1.0). Requires spacy-transformers (installed automatically). Built for SpaCy 3.4.x — installed with --no-deps on newer versions.",
        extra_packages = ("spacy-transformers",),
    ),

    # ── German ───────────────────────────────────────────────────────────
    SpacyModelInfo(
        name        = "de_core_news_sm",
        language    = "German",
        lang_code   = "de",
        size        = "sm",
        ram_mb      = 15,
        description = "Small · News corpus · ~15 MB",
        recommended = True,
    ),
    SpacyModelInfo(
        name        = "de_core_news_md",
        language    = "German",
        lang_code   = "de",
        size        = "md",
        ram_mb      = 50,
        description = "Medium · Word vectors · ~50 MB",
    ),
    SpacyModelInfo(
        name        = "de_core_news_lg",
        language    = "German",
        lang_code   = "de",
        size        = "lg",
        ram_mb      = 560,
        description = "Large · Full vectors · ~560 MB",
    ),
    SpacyModelInfo(
        name        = "de_dep_news_trf",
        language    = "German",
        lang_code   = "de",
        size        = "trf",
        ram_mb      = 450,
        description = "Transformer · Best German accuracy · Requires GPU for speed",
    ),

    # ── French ───────────────────────────────────────────────────────────
    SpacyModelInfo(
        name        = "fr_core_news_sm",
        language    = "French",
        lang_code   = "fr",
        size        = "sm",
        ram_mb      = 15,
        description = "Small · News corpus · ~15 MB",
        recommended = True,
    ),
    SpacyModelInfo(
        name        = "fr_core_news_md",
        language    = "French",
        lang_code   = "fr",
        size        = "md",
        ram_mb      = 50,
        description = "Medium · Word vectors · ~50 MB",
    ),
    SpacyModelInfo(
        name        = "fr_core_news_lg",
        language    = "French",
        lang_code   = "fr",
        size        = "lg",
        ram_mb      = 560,
        description = "Large · Full vectors · ~560 MB",
    ),
    SpacyModelInfo(
        name        = "fr_dep_news_trf",
        language    = "French",
        lang_code   = "fr",
        size        = "trf",
        ram_mb      = 450,
        description = "Transformer · Best French accuracy · Requires GPU for speed",
    ),

    # ── Spanish ──────────────────────────────────────────────────────────
    SpacyModelInfo(
        name        = "es_core_news_sm",
        language    = "Spanish",
        lang_code   = "es",
        size        = "sm",
        ram_mb      = 13,
        description = "Small · News corpus · ~13 MB",
        recommended = True,
    ),
    SpacyModelInfo(
        name        = "es_core_news_md",
        language    = "Spanish",
        lang_code   = "es",
        size        = "md",
        ram_mb      = 50,
        description = "Medium · Word vectors · ~50 MB",
    ),
    SpacyModelInfo(
        name        = "es_core_news_lg",
        language    = "Spanish",
        lang_code   = "es",
        size        = "lg",
        ram_mb      = 560,
        description = "Large · Full vectors · ~560 MB",
    ),
    SpacyModelInfo(
        name        = "es_dep_news_trf",
        language    = "Spanish",
        lang_code   = "es",
        size        = "trf",
        ram_mb      = 450,
        description = "Transformer · Best Spanish accuracy · Requires GPU for speed",
    ),

    # ── Italian ──────────────────────────────────────────────────────────
    SpacyModelInfo(
        name        = "it_core_news_sm",
        language    = "Italian",
        lang_code   = "it",
        size        = "sm",
        ram_mb      = 13,
        description = "Small · News corpus · ~13 MB",
        recommended = True,
    ),
    SpacyModelInfo(
        name        = "it_core_news_md",
        language    = "Italian",
        lang_code   = "it",
        size        = "md",
        ram_mb      = 50,
        description = "Medium · Word vectors · ~50 MB",
    ),
    SpacyModelInfo(
        name        = "it_core_news_lg",
        language    = "Italian",
        lang_code   = "it",
        size        = "lg",
        ram_mb      = 560,
        description = "Large · Full vectors · ~560 MB",
    ),

    # ── Dutch ────────────────────────────────────────────────────────────
    SpacyModelInfo(
        name        = "nl_core_news_sm",
        language    = "Dutch",
        lang_code   = "nl",
        size        = "sm",
        ram_mb      = 13,
        description = "Small · News corpus · ~13 MB",
        recommended = True,
    ),
    SpacyModelInfo(
        name        = "nl_core_news_md",
        language    = "Dutch",
        lang_code   = "nl",
        size        = "md",
        ram_mb      = 50,
        description = "Medium · Word vectors · ~50 MB",
    ),
    SpacyModelInfo(
        name        = "nl_core_news_lg",
        language    = "Dutch",
        lang_code   = "nl",
        size        = "lg",
        ram_mb      = 560,
        description = "Large · Full vectors · ~560 MB",
    ),

    # ── Portuguese ───────────────────────────────────────────────────────
    SpacyModelInfo(
        name        = "pt_core_news_sm",
        language    = "Portuguese",
        lang_code   = "pt",
        size        = "sm",
        ram_mb      = 13,
        description = "Small · News corpus · ~13 MB",
        recommended = True,
    ),
    SpacyModelInfo(
        name        = "pt_core_news_lg",
        language    = "Portuguese",
        lang_code   = "pt",
        size        = "lg",
        ram_mb      = 560,
        description = "Large · Full vectors · ~560 MB",
    ),
]


def get_spacy_model(name: str) -> SpacyModelInfo | None:
    """Return model info by name; ``None`` if not in catalog."""
    for m in SPACY_CATALOG:
        if m.name == name:
            return m
    return None


def get_models_by_language(lang_code: str) -> list[SpacyModelInfo]:
    """Return all catalog models for a given language code (e.g. ``"tr"``)."""
    return [m for m in SPACY_CATALOG if m.lang_code == lang_code]


def recommended_for_language(lang_code: str) -> SpacyModelInfo | None:
    """Return the recommended model for a language; ``None`` if unsupported."""
    for m in SPACY_CATALOG:
        if m.lang_code == lang_code and m.recommended:
            return m
    return None

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from wardcat._entity_policy import EntityPolicyMixin
from wardcat.config.loader import load_config
from wardcat.core.engine import DetectionEngine
from wardcat.core.models import KNOWN_ENTITY_TYPES, ScanResult
from wardcat.core.registry import (
    GLINER_ENTITIES,
    LAYER_ENTITIES,
    NER_ENTITIES,
    REGEX_ENTITIES,
    VALID_LAYERS,
)
from wardcat.detectors.base import BaseDetector
from wardcat.detectors.regex_detector import RegexDetector
from wardcat.exceptions import ConfigError, UnsupportedLanguageError
from wardcat.llm.backends.base import Backend
from wardcat.ner.spacy_catalog import Language

logger = logging.getLogger(__name__)


def _resolve_spacy_model(model: str) -> str:
    """Suggests an alternative if the requested SpaCy model is not installed.

    Behavior:
    - Returns the model as-is if it is installed.
    - If not installed, lists available SpaCy models and logs a warning.
    - If no model is found at all, returns the original name (NERDetector will raise its own error).

    Thread safety: uses spacy.util.get_installed_models() (read-only registry lookup)
    rather than spacy.load() to avoid a redundant full model load — NERDetector
    already loads and caches the model under its own lock (_CACHE_LOCK).
    """
    try:
        import spacy.util

        installed = list(spacy.util.get_installed_models())
    except Exception:
        installed = []

    if model in installed:
        logger.info("SpaCy model loaded: %r", model)
        return model

    if not installed:
        logger.warning(
            "SpaCy model %r is not installed and no models are available. "
            "Install with: python -m spacy download %s",
            model,
            model,
        )
        return model

    # Match by language prefix (tr_, en_, etc.)
    lang_prefix = model.split("_")[0] + "_"
    same_lang = [m for m in installed if m.startswith(lang_prefix)]
    fallback = same_lang[0] if same_lang else installed[0]

    logger.warning(
        "SpaCy model %r is not installed — falling back to %r.\n"
        "  Installed models: %s\n"
        "  To install the correct model: python -m spacy download %s",
        model,
        fallback,
        installed,
        model,
    )
    return fallback


class Wardcat(EntityPolicyMixin):
    """
    The main interface exposed to users.

    Programmatic API (method chaining)::

        import os
        from wardcat import Wardcat, Entity, Action

        guard = (
            # Read secrets from the environment in YOUR app — the library itself
            # never reads env vars; pass everything explicitly.
            Wardcat(salt=os.environ["WARDCAT_SALT"])
            .add_entity(Entity.EMAIL,       action=Action.HASH)
            .add_entity(Entity.CREDIT_CARD, action=Action.HASH)
            .remove_entity(Entity.ORG)
        )
        result = guard.scan(text)

    Enable everything, then prune::

        guard = Wardcat(salt="...").add_entity(Entity.ALL, action="hash")
        guard.remove_entity(Entity.ORG)
        guard.entity_policy()   # inspect: {"CREDIT_CARD": "hash", ...}

    Declarative API (YAML)::

        guard = Wardcat(config_path="config/my_policy.yaml")
        result = guard.scan(text)

    Configuration is explicit: pass constructor arguments or a YAML ``config_path``.
    The library does **not** read environment variables — read any secrets in your
    own application and hand them to the constructor.

    LLM detector — configured with :meth:`with_llm` (or a YAML ``config_path``),
    not constructor arguments::

        guard = Wardcat(salt="s").with_llm(model="llama3.1:8b")
        result = guard.scan(text)

    NER requires an explicit model — wardcat ships no default. Choose one in a
    documented way via ``language=`` (recommended) or ``spacy_model=``::

        from wardcat import Wardcat, Language

        guard = Wardcat(language=Language.DE, spacy_size="md")   # → de_core_news_md
        guard = Wardcat(language=[Language.DE, Language.FR])     # one model per language
        guard = Wardcat(spacy_model="en_core_web_sm")           # explicit package name
        guard = Wardcat(spacy_model=["en_core_web_sm", "de_core_news_sm"])  # multiple
        # Supported languages: en, de, fr, es, it, nl, pt, tr; sizes sm/md/lg/trf.
        # A named-but-missing model is auto-downloaded (spacy_auto_download=False to disable).
        # ``use_ner=True`` without a model (or language) raises ConfigError.
        # For mixed-language text without extra models, use the LLM layer (use_llm=True),
        # whose prompt is multilingual.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        salt: str = "",
        use_ner: bool | None = None,  # None → inherit config (off unless YAML/model says on)
        spacy_model: str | list[str] | None = None,  # explicit SpaCy model name(s)
        language: str | Language | list[str | Language] | None = None,  # NER by language
        spacy_size: str = "sm",  # NER: model size tier when language is given (sm/md/lg/trf)
        spacy_auto_download: bool | None = None,  # download the SpaCy model if missing
    ) -> None:
        self._config = load_config(config_path)

        # Constructor arguments override YAML
        if salt:
            self._config["salt"] = salt

        # ── NER model selection ────────────────────────────────────────────
        # wardcat ships no default SpaCy model: NER requires an explicit choice,
        # made in a documented way via `language=` (recommended, supports a list
        # for multilingual NER) or `spacy_model=` (explicit package name(s)).
        # Specifying either resolves the model(s) and implies NER on (unless the
        # caller explicitly passed use_ner=False); a named-but-missing model is
        # auto-downloaded.
        specified_model = language is not None or spacy_model is not None
        if language is not None:
            models = self._resolve_language_models(language, spacy_size)
            self._config["spacy_models"] = models
            self._config["spacy_model"] = models[0]
            if spacy_auto_download is None:
                spacy_auto_download = True
        if spacy_model is not None:
            names = [spacy_model] if isinstance(spacy_model, str) else list(spacy_model)
            names = list(dict.fromkeys(names))  # dedupe, preserve order
            if not names:
                raise ConfigError("spacy_model is empty — pass at least one model name.")
            self._config["spacy_models"] = names
            self._config["spacy_model"] = names[0]
            if spacy_auto_download is None:
                spacy_auto_download = True

        # Effective NER flag: an explicit constructor value wins; otherwise NER is
        # on when a model/language was specified, else it inherits the config
        # (YAML/default, which is off).
        if use_ner is None:
            ner_on = specified_model or bool(self._config.get("use_ner", False))
        else:
            ner_on = use_ner
        self._config["use_ner"] = ner_on
        if spacy_auto_download:
            self._config["spacy_auto_download"] = True

        # NER on but no model resolved (constructor or YAML) → hard error.
        if ner_on and not (self._config.get("spacy_models") or self._config.get("spacy_model")):
            raise ConfigError(
                "use_ner=True requires a SpaCy model, but none was given. "
                "Pass language=... (e.g. Language.EN, or a list for multilingual NER) "
                "or spacy_model=...; wardcat ships no default model."
            )

        # The LLM detector is configured via with_llm() or a YAML config_path —
        # not constructor arguments. Ensure the sub-config exists for YAML/builder use.
        self._config.setdefault("llm_detector", {})

        # Warn at most once when a hash action is active without a salt. Checked
        # in _rebuild() too, since entities are opt-in and usually added after init.
        self._salt_warned = False
        self._default_action_warned = False
        self._rebuild()

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan(self, text: str) -> ScanResult:
        """Scan text and return a ScanResult."""
        return self._engine.scan(text)

    async def scan_async(self, text: str) -> ScanResult:
        """Async scan — uses native async I/O for the LLM backend when available.

        CPU-bound detectors (regex, SpaCy NER) run in a thread pool;
        the LLM detector (if enabled) uses ``httpx.AsyncClient`` natively,
        so multiple concurrent calls do not block each other.
        """
        return await self._engine.scan_async(text)

    def scan_batch(self, texts: list[str], *, max_workers: int | None = None) -> list[ScanResult]:
        """
        Scan multiple texts in parallel using a thread pool.

        Each text is scanned independently; an error in a single item does
        not affect the others — the original text is returned untouched
        for any item that fails.

        :param texts:       List of texts to scan
        :param max_workers: Number of parallel threads. Defaults to the
                            ``scan_batch_workers`` config value (default: 4).
        :returns:           List of ``ScanResult`` in the same order as ``texts``
        """
        if not texts:
            return []

        workers = max_workers or self._config.get("scan_batch_workers", 4)

        results: list[ScanResult | None] = [None] * len(texts)

        def _scan_one(idx: int, text: str) -> tuple[int, ScanResult]:
            try:
                return idx, self._engine.scan(text)
            except Exception as exc:
                logger.error(
                    "scan_batch item %d failed (%s: %s), returning original text.",
                    idx,
                    type(exc).__name__,
                    exc,
                )
                return idx, ScanResult(
                    original_text=text,
                    sanitized_text=text,
                    violations=[],
                    scan_error=f"{type(exc).__name__}: {exc}",
                )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_scan_one, i, text): i for i, text in enumerate(texts)}
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results  # type: ignore[return-value]

    async def scan_batch_async(
        self, texts: list[str], *, max_workers: int | None = None
    ) -> list[ScanResult]:
        """Scan multiple texts concurrently using native async.

        Each text is scanned independently via :meth:`scan_async`; all are
        run concurrently with ``asyncio.gather``.  Errors in individual items
        are caught — the original text is returned with ``scan_error`` set.
        """
        if not texts:
            return []

        async def _one(idx: int, text: str) -> tuple[int, ScanResult]:
            try:
                return idx, await self.scan_async(text)
            except Exception as exc:
                logger.error(
                    "scan_batch_async item %d failed (%s: %s), returning original text.",
                    idx,
                    type(exc).__name__,
                    exc,
                )
                return idx, ScanResult(
                    original_text=text,
                    sanitized_text=text,
                    violations=[],
                    scan_error=f"{type(exc).__name__}: {exc}",
                )

        pairs = await asyncio.gather(*(_one(i, t) for i, t in enumerate(texts)))
        results: list[ScanResult | None] = [None] * len(texts)
        for idx, result in pairs:
            results[idx] = result
        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Discoverability
    # (entity add/remove/change + introspection live in EntityPolicyMixin)
    # ------------------------------------------------------------------
    @staticmethod
    def supported_entities(layer: str | None = None) -> frozenset[str]:
        """Return the entity types wardcat can detect (discoverability helper).

        ::

            Wardcat.supported_entities()            # every known entity type
            Wardcat.supported_entities("regex")     # only what the regex layer detects
            Wardcat.supported_entities("ner")       # PERSON, ORG, ADDRESS
            Wardcat.supported_entities("gliner")    # zero-shot NER (PII types)
            Wardcat.supported_entities("llm")       # contextual/semantic types

        :param layer: ``None`` → all known types; or one of ``"regex"``,
                      ``"ner"``, ``"gliner"``, ``"llm"`` for that layer's set.
        :raises ConfigError: if ``layer`` is not a known layer.
        """
        if layer is None:
            return frozenset(KNOWN_ENTITY_TYPES)
        if layer not in LAYER_ENTITIES:
            raise ConfigError(f"Unknown layer {layer!r}. Valid layers: {sorted(VALID_LAYERS)}")
        return LAYER_ENTITIES[layer]

    # ------------------------------------------------------------------
    # Layer builders (fluent alternative to the constructor's llm_*/spacy_* args)
    # ------------------------------------------------------------------

    def with_ner(
        self,
        *,
        language: str | Language | list[str | Language] | None = None,
        spacy_model: str | list[str] | None = None,
        spacy_size: str = "sm",
        auto_download: bool = True,
    ) -> Wardcat:
        """
        Enable the SpaCy NER layer with an explicit model. Supports chaining.

        Mirrors :meth:`with_llm`. Pass ``language=`` (recommended; a list enables
        multilingual NER) or ``spacy_model=`` (explicit package name(s)).

        ::

            guard = Wardcat(salt="s").with_ner(language=Language.EN)
            guard = Wardcat(salt="s").with_ner(spacy_model=["en_core_web_sm", "de_core_news_sm"])

        :raises ConfigError: if neither ``language`` nor ``spacy_model`` is given.
        """
        if language is None and spacy_model is None:
            raise ConfigError(
                "with_ner() requires a model — pass language=... (e.g. Language.EN) "
                "or spacy_model=...; wardcat ships no default model."
            )
        if language is not None:
            models = self._resolve_language_models(language, spacy_size)
        else:
            models = [spacy_model] if isinstance(spacy_model, str) else list(spacy_model)  # type: ignore[arg-type]
            models = list(dict.fromkeys(models))
            if not models:
                raise ConfigError("spacy_model is empty — pass at least one model name.")
        self._config["spacy_models"] = models
        self._config["spacy_model"] = models[0]
        self._config["use_ner"] = True
        if auto_download:
            self._config["spacy_auto_download"] = True
        self._rebuild()
        return self

    def with_llm(
        self,
        *,
        backend: str | Backend = Backend.OLLAMA,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        api_key: str = "",
        timeout: int = 60,
        allow_http: bool = False,
        adjudicate: bool = False,
        auto_pull: bool = False,
        device_map: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> Wardcat:
        """
        Enable the on-prem LLM detector. Supports chaining; mirrors :meth:`with_ner`.

        A fluent alternative to the constructor's ``llm_*`` arguments — keeps the
        LLM configuration in one place::

            guard = (
                Wardcat(salt="s")
                .with_ner(language=Language.TR)
                .with_llm(backend=Backend.OLLAMA, model="llama3.2", adjudicate=True)
            )

        ``backend`` is the backend *type* (:class:`~wardcat.Backend`); the
        *address* goes to ``base_url``.
        """
        llm_cfg = self._config.setdefault("llm_detector", {})
        llm_cfg.update(
            {
                "enabled": True,
                "backend": backend.value if isinstance(backend, Backend) else backend,
                "model": model,
                "base_url": base_url,
                "api_key": api_key,
                "timeout": timeout,
                "allow_http": allow_http,
                "adjudicate": adjudicate,
                "auto_pull": auto_pull,
                "device_map": device_map,
                "load_in_8bit": load_in_8bit,
                "load_in_4bit": load_in_4bit,
            }
        )
        self._rebuild()
        return self

    def with_gliner(
        self,
        *,
        model: str = "fastino/gliner2-privacy-filter-PII-multi",
        threshold: float = 0.5,
        quantize: bool = False,
        chunk_size: int = 1500,
    ) -> Wardcat:
        """
        Enable the GLiNER zero-shot NER layer. Supports chaining; mirrors
        :meth:`with_ner`.

        GLiNER is a lightweight transformer NER that sits between SpaCy NER and
        the LLM: it detects prompt-given entity types with better context than
        SpaCy, far cheaper than an LLM. It runs *alongside* SpaCy when both are
        enabled — the engine merges their spans and a checksum-validated regex
        span always wins an overlap.

        Like NER, the entity types it scans for are **opt-in** — enabling the
        layer alone detects nothing; add the entities you want::

            guard = (
                Wardcat(salt="s")
                .with_gliner()                 # GLiNER layer on
                .add_entity(Entity.PERSON, Action.HASH)
                .add_entity(Entity.EMAIL, Action.REDACT)
            )

        Needs the ``gliner`` extra (``pip install "wardcat[gliner]"``). The
        default model is multilingual (EN/FR/ES/DE/IT/PT/NL) but **not Turkish**
        — keep the regex/LLM layers for Turkish text.

        :param model:      GLiNER2 model id (HuggingFace).
        :param threshold:  drop detected spans below this confidence (0–1).
        :param quantize:   load a quantized model (less memory, slightly lower quality).
        :param chunk_size: split longer input into overlapping windows of this many
                           characters — GLiNER truncates long text, so this keeps it
                           from silently missing PII late in a document.
        """
        self._config["gliner_detector"] = {
            "enabled": True,
            "model": model,
            "threshold": threshold,
            "quantize": quantize,
            "chunk_size": chunk_size,
        }
        self._rebuild()
        return self

    # ------------------------------------------------------------------
    # NER model resolution (build helper)
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_language_models(
        language: str | Language | list[str | Language], spacy_size: str
    ) -> list[str]:
        """Resolve one or more languages (+ size tier) to concrete SpaCy model names."""
        from wardcat.ner.spacy_catalog import (
            SPACY_CATALOG,
            get_models_by_language,
            resolve_model,
        )

        items = [language] if isinstance(language, str) else list(language)
        models: list[str] = []
        for lang in items:
            code = (lang.value if isinstance(lang, Language) else str(lang)).lower()
            info = resolve_model(code, spacy_size)
            if info is None:
                if get_models_by_language(code):
                    raise UnsupportedLanguageError(
                        f"No compatible SpaCy model for language {code!r} "
                        f"at size {spacy_size!r}. Try a different size (sm/md/lg)."
                    )
                supported = sorted({m.lang_code for m in SPACY_CATALOG})
                raise UnsupportedLanguageError(
                    f"Unsupported language {code!r}. Supported: {supported}."
                )
            models.append(info.name)
        return list(dict.fromkeys(models))  # dedupe, preserve order

    def set_salt(self, salt: str) -> Wardcat:
        """Update the hash salt."""
        self._config["salt"] = salt
        self._rebuild()
        return self

    def add_allowlist(self, values: list[str]) -> Wardcat:
        """Add exact values that should never be flagged as PII.

        Supports method chaining::

            guard.add_allowlist(["no-reply@company.com", "192.168.1.1"])

        :param values: List of exact string values to exempt from detection.
        """
        existing: list[str] = self._config.setdefault("allowlist", [])
        for v in values:
            if v not in existing:
                existing.append(v)
        self._rebuild()
        return self

    def add_denylist(self, entries: list[dict[str, str]]) -> Wardcat:
        """Add values that should always be flagged as PII.

        Each entry must have a ``value`` key and an ``entity_type`` key.
        The action applied is taken from the entity's config (same as
        regular detections).  Supports method chaining::

            guard.add_denylist([
                {"value": "John Smith",    "entity_type": "PERSON"},
                {"value": "ProjectSecret", "entity_type": "CUSTOM_SECRET"},
            ])

        :param entries: List of dicts with ``value`` and ``entity_type`` keys.
        """
        existing: list[dict[str, str]] = self._config.setdefault("denylist", [])
        for entry in entries:
            if not isinstance(entry, dict):
                raise ConfigError(
                    f"Each denylist entry must be a dict with a 'value' or 'pattern' key: {entry!r}"
                )
            if "value" not in entry and "pattern" not in entry:
                raise ConfigError(
                    f"Each denylist entry must have either a 'value' or a 'pattern' key: {entry!r}"
                )
            existing.append(entry)
        self._rebuild()
        return self

    def with_propagation(self, *, enabled: bool = True, min_length: int = 3) -> Wardcat:
        """Redact **every** occurrence of a value once any layer detects it.

        Model-based layers (GLiNER/NER/LLM) often report a repeated value only
        once, which would leave the other occurrences unredacted. With
        propagation on, a value detected anywhere is anonymized at every
        whole-token occurrence in the text — using that value's entity type and
        action. Deterministic regex spans still win overlaps, so a propagated
        match never displaces a checksum-validated one. Chainable::

            guard = Wardcat(salt="s").with_gliner().add_entity("PERSON").with_propagation()

        It can over-redact (e.g. a short common name), so it is **off by default**
        and only exact, token-bounded matches at least ``min_length`` chars long
        are propagated.

        :param enabled:    turn propagation on (default) or off.
        :param min_length: skip values shorter than this many characters.
        """
        self._config["propagate_matches"] = enabled
        self._config["propagate_min_length"] = min_length
        self._rebuild()
        return self

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_warn_unsalted(self) -> None:
        """Warn once if a hash action is active but no salt is set."""
        if self._salt_warned or self._config.get("salt"):
            return
        entity_cfg = self._config.get("entities", {})
        has_hash = any(
            isinstance(cfg, dict) and cfg.get("action") == "hash" and cfg.get("enabled")
            for cfg in entity_cfg.values()
        )
        if has_hash:
            self._salt_warned = True
            logger.warning(
                "No hash salt set — using unsalted hashes (identical PII always yields the "
                "same hash, leaving them open to rainbow-table attacks). Pass salt=... or set "
                "the WARDCAT_SALT environment variable in production."
            )

    def _rebuild(self) -> None:
        """Rebuild detectors and engine when configuration changes."""
        self._maybe_warn_unsalted()
        self._detectors: list[BaseDetector] = []
        entity_cfg = self._config.get("entities", {})

        # Regex detector
        custom_patterns = self._config.get("custom_patterns", {})
        # Register custom pattern actions in entities config so the engine can look them up
        for cp_name, cp_cfg in custom_patterns.items():
            entity_cfg.setdefault(
                cp_name, {"enabled": True, "action": cp_cfg.get("action", "warn")}
            )
        # Entities are opt-in: only those explicitly enabled (via add_entity / YAML)
        # run. An entity absent from the config is OFF.
        enabled_regex = {e for e in REGEX_ENTITIES if entity_cfg.get(e, {}).get("enabled", False)}
        if enabled_regex or custom_patterns:
            self._detectors.append(
                RegexDetector(
                    enabled_regex,
                    custom_patterns=custom_patterns,
                    fold_confusables_enabled=self._config.get("normalize_confusables", True),
                )
            )

        # SpaCy NER detector(s) (optional). A list of models loads one detector
        # per language (multilingual NER); the engine merges their spans.
        if self._config.get("use_ner", True):
            enabled_ner = {e for e in NER_ENTITIES if entity_cfg.get(e, {}).get("enabled", False)}
            if enabled_ner:
                models = self._config.get("spacy_models") or (
                    [self._config["spacy_model"]] if self._config.get("spacy_model") else []
                )
                auto_download = self._config.get("spacy_auto_download", False)
                loaded: set[str] = set()
                for model in models:
                    # Each model is loaded independently — one failure must not
                    # disable the others.
                    try:
                        from wardcat.detectors.ner_detector import NERDetector

                        if auto_download:
                            from wardcat.ner.downloader import ensure_model

                            ensure_model(model, auto_download=True)
                        resolved = _resolve_spacy_model(model)
                        if resolved in loaded:  # avoid duplicate detectors
                            continue
                        loaded.add(resolved)
                        self._detectors.append(NERDetector(enabled_ner, resolved))
                    except Exception as exc:
                        logger.warning(
                            "SpaCy NER model %r could not be loaded, skipping it. Error: %s",
                            model,
                            exc,
                        )

        # GLiNER detector (optional). A zero-shot transformer NER; runs alongside
        # SpaCy NER when both are on. Entity types are opt-in (shared with the
        # regex/NER policy), so the layer only fires for enabled entities it
        # supports. Loaded lazily and skipped (with a warning) if the optional
        # gliner dependency or model is unavailable.
        gliner_cfg = self._config.get("gliner_detector", {})
        if gliner_cfg.get("enabled", False):
            enabled_gliner = {
                e for e in GLINER_ENTITIES if entity_cfg.get(e, {}).get("enabled", False)
            }
            if enabled_gliner:
                try:
                    from wardcat.detectors.gliner_detector import GLiNERDetector

                    self._detectors.append(
                        GLiNERDetector(
                            enabled_gliner,
                            model=gliner_cfg.get(
                                "model", "fastino/gliner2-privacy-filter-PII-multi"
                            ),
                            threshold=gliner_cfg.get("threshold", 0.5),
                            quantize=gliner_cfg.get("quantize", False),
                            chunk_size=gliner_cfg.get("chunk_size", 1500),
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "GLiNER model %r could not be loaded, skipping it. "
                        "Install the extra with: pip install 'wardcat[gliner]'. Error: %s",
                        gliner_cfg.get("model"),
                        exc,
                    )

        # LLM detector (optional)
        llm_cfg = self._config.get("llm_detector", {})
        if llm_cfg.get("enabled", False):
            self._detectors.append(self._build_llm_detector(llm_cfg))

        self._engine = DetectionEngine(self._config, self._detectors)

    def _build_llm_detector(self, llm_cfg: dict[str, Any]) -> BaseDetector:
        """Build the LLM detector according to configuration."""
        from wardcat.detectors.llm_detector import LLMDetector
        from wardcat.llm.backends.registry import create_backend

        # The backend is built from the registry — adding a backend needs no
        # change here (wardcat.register_backend(name, factory)).
        backend = create_backend(llm_cfg)
        timeout = llm_cfg.get("timeout", 60)

        entity_cfg = llm_cfg.get("entities", {})
        enabled = {e for e, cfg in entity_cfg.items() if cfg.get("enabled", True)}

        # Make the LLM entities' *actions* available to the engine for applying to
        # LLM-detected spans — but as enabled=False so they do NOT switch on the
        # regex/NER layer for the same entity (the LLM layer is opt-in on its own).
        for entity, cfg in entity_cfg.items():
            self._config["entities"].setdefault(
                entity, {"enabled": False, "action": cfg.get("action", "warn")}
            )

        cache_ttl = llm_cfg.get("cache_ttl", 0)
        return LLMDetector(
            backend=backend, enabled_entities=enabled, timeout=timeout, cache_ttl=cache_ttl
        )

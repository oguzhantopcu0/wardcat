from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ai_guard.config.loader import load_config
from ai_guard.core.engine import DetectionEngine
from ai_guard.core.models import (
    KNOWN_ENTITY_TYPES,
    Action,
    Entity,
    ScanResult,
    warn_unknown_entity,
)
from ai_guard.detectors.base import BaseDetector
from ai_guard.detectors.regex_detector import RegexDetector
from ai_guard.exceptions import ConfigError, UnsupportedLanguageError
from ai_guard.llm.backends.base import Backend
from ai_guard.ner.spacy_catalog import Language

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
            "Install with: python -m spacy download %s  "
            "or: python -m ai_guard spacy download %s",
            model,
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
        "  To install the correct model: python -m ai_guard spacy download %s",
        model,
        fallback,
        installed,
        model,
    )
    return fallback


# Central table mapping each entity to its detector
_REGEX_ENTITIES = {
    "CREDIT_CARD",
    "EMAIL",
    "PHONE",
    "IBAN",
    "IP_ADDRESS",
    "IPv6",
    "TC_ID",
    "ADDRESS",
    "POSTAL_CODE",
    "UUID",
    "SSN",
    "MAC_ADDRESS",
    "JWT",
    "NIN",
    "CUSTOM_SECRET",
    "UK_POSTAL_CODE",
    "US_ZIP_CODE",
    "EU_NATIONAL_ID",
    "PASSPORT",
    "CODICE_FISCALE",
    "DATE_OF_BIRTH",
    "VEHICLE_PLATE",
    "FINANCIAL_AMOUNT",
    "VAT_NUMBER",
}
_NER_ENTITIES = {"PERSON", "ORG", "ADDRESS"}

# Entity types the LLM layer can be asked to detect (those with prompt guidance).
from ai_guard.llm.prompt import SUPPORTED_ENTITIES as _LLM_ENTITIES  # noqa: E402

# Detector layers a filter can be applied to, and which entities each supports.
_LAYER_ENTITIES: dict[str, frozenset[str]] = {
    "regex": frozenset(_REGEX_ENTITIES),
    "ner": frozenset(_NER_ENTITIES),
    "llm": _LLM_ENTITIES,
}
_VALID_LAYERS = frozenset(_LAYER_ENTITIES)


class AIGuard:
    """
    The main interface exposed to users.

    Programmatic API (method chaining)::

        import os
        from ai_guard import AIGuard, Entity, Action

        guard = (
            # Read secrets from the environment in YOUR app — the library itself
            # never reads env vars; pass everything explicitly.
            AIGuard(salt=os.environ["AIGUARD_SALT"])
            .add_entity(Entity.EMAIL,       action=Action.HASH)
            .add_entity(Entity.CREDIT_CARD, action=Action.HASH)
            .remove_entity(Entity.ORG)
        )
        result = guard.scan(text)

    Enable everything, then prune::

        guard = AIGuard(salt="...").add_entity(Entity.ALL, action="hash")
        guard.remove_entity(Entity.ORG)
        guard.entity_policy()   # inspect: {"CREDIT_CARD": "hash", ...}

    Declarative API (YAML)::

        guard = AIGuard(config_path="config/my_policy.yaml")
        result = guard.scan(text)

    Configuration is explicit: pass constructor arguments or a YAML ``config_path``.
    The library does **not** read environment variables (the ``ai-guard`` CLI does,
    as it is an application — see ``AIGUARD_*`` in ``python -m ai_guard --help``).

    LLM detector (Ollama)::

        guard = AIGuard(
            use_llm=True,
            llm_model="llama3.1:8b",
            llm_base_url="http://localhost:11434",
        )
        result = guard.scan(text)

    NER requires an explicit model — ai-guard ships no default. Choose one in a
    documented way via ``language=`` (recommended) or ``spacy_model=``::

        from ai_guard import AIGuard, Language

        guard = AIGuard(language=Language.DE, spacy_size="md")   # → de_core_news_md
        guard = AIGuard(language=[Language.DE, Language.FR])     # one model per language
        guard = AIGuard(spacy_model="en_core_web_sm")           # explicit package name
        guard = AIGuard(spacy_model=["en_core_web_sm", "de_core_news_sm"])  # multiple
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
        use_llm: bool = False,
        llm_backend: str | Backend = "ollama",  # Backend.OLLAMA | OPENAI_COMPATIBLE | TRANSFORMERS
        llm_model: str = "llama3.2",
        llm_base_url: str = "http://localhost:11434",
        llm_api_key: str = "",
        llm_timeout: int = 60,
        llm_allow_http: bool = False,  # allow plaintext HTTP to a remote LLM (not recommended)
        llm_adjudicate: bool = False,  # ensemble: LLM verifies regex/NER candidates
        auto_pull: bool = False,  # Ollama: automatically download if model is missing
        llm_device_map: str = "auto",  # Transformers: GPU distribution
        llm_load_in_8bit: bool = False,  # Transformers: 8-bit quantization
        llm_load_in_4bit: bool = False,  # Transformers: 4-bit quantization
    ) -> None:
        self._config = load_config(config_path)

        # Constructor arguments override YAML
        if salt:
            self._config["salt"] = salt

        # ── NER model selection ────────────────────────────────────────────
        # ai-guard ships no default SpaCy model: NER requires an explicit choice,
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
                "or spacy_model=...; ai-guard ships no default model."
            )

        # LLM detector overrides
        llm_cfg = self._config.setdefault("llm_detector", {})
        if use_llm:
            llm_cfg["enabled"] = True
        if llm_backend != "ollama":
            llm_cfg["backend"] = (
                llm_backend.value if isinstance(llm_backend, Backend) else llm_backend
            )
        if llm_model != "llama3.2":
            llm_cfg["model"] = llm_model
        if llm_base_url != "http://localhost:11434":
            llm_cfg["base_url"] = llm_base_url
        if llm_api_key:
            llm_cfg["api_key"] = llm_api_key
        if llm_timeout != 60:
            llm_cfg["timeout"] = llm_timeout
        if llm_allow_http:
            llm_cfg["allow_http"] = True
        if llm_adjudicate:
            llm_cfg["adjudicate"] = True
        if auto_pull:
            llm_cfg["auto_pull"] = True
        if llm_device_map != "auto":
            llm_cfg["device_map"] = llm_device_map
        if llm_load_in_8bit:
            llm_cfg["load_in_8bit"] = True
        if llm_load_in_4bit:
            llm_cfg["load_in_4bit"] = True

        # Warn at most once when a hash action is active without a salt. Checked
        # in _rebuild() too, since entities are opt-in and usually added after init.
        self._salt_warned = False
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
    # Programmatic API
    # ------------------------------------------------------------------

    def add_entity(
        self,
        entity_type: str | Entity,
        action: str | Action | None = None,
        layers: list[str] | None = None,
    ) -> AIGuard:
        """
        Enable a single entity type (or every type via :attr:`Entity.ALL`).

        Adding an entity always enables it — to turn one off, use
        :meth:`remove_entity`. Supports method chaining.

        :param entity_type: An :class:`~ai_guard.Entity` constant (recommended,
                            e.g. ``Entity.EMAIL``) or its string form (``"EMAIL"``).
                            Pass :attr:`Entity.ALL` to enable **every** known
                            entity in one call, then prune with
                            :meth:`remove_entity`.
        :param action:      An :class:`~ai_guard.Action` constant (e.g. ``Action.HASH``)
                            or its string form: ``"warn"``, ``"hash"``, ``"redact"``, ``"mask"``.
                            **When omitted, defaults to** ``"hash"`` (the safest
                            action) and a warning is logged — pass ``action``
                            explicitly to silence it.
        :param layers:      Which detector layers should look for this entity —
                            any of ``"regex"``, ``"ner"``, ``"llm"``. When
                            ``None`` (default), every layer that supports the
                            entity is used. Use this to target one layer, e.g.
                            ``layers=["llm"]`` for contextual/semantic entities.
        :raises ConfigError: if ``entity_type`` is not a ``str``/``Entity``, the
                            ``action`` is invalid, or a ``layer`` is unknown.
        """
        action = self._action_or_default(action)
        if self._is_all(entity_type):
            for name in sorted(KNOWN_ENTITY_TYPES):
                self._set_entity(name, enabled=True, action=action, layers=layers)
        else:
            self._set_entity(entity_type, enabled=True, action=action, layers=layers)
        self._rebuild()
        return self

    def add_entities(
        self,
        entities,
        *,
        action: str | Action | None = None,
        layers: list[str] | None = None,
    ) -> AIGuard:
        """
        Enable many entity types at once (single rebuild). Supports chaining.

        Like :meth:`add_entity`, every listed entity is enabled; use
        :meth:`remove_entities` to turn entities off. ``entities`` may be:

        * an iterable of names — applied with the given ``action``/``layers``::

              from ai_guard import turkish_entities
              guard.add_entities(turkish_entities(), action="hash")
              guard.add_entities(["EMAIL", "CREDIT_CARD"], action="hash")

        * a mapping ``{name: action}``::

              guard.add_entities({"EMAIL": "warn", "CREDIT_CARD": "hash"})

        * a mapping ``{name: {"action": ..., "layers": ...}}`` for per-entity
          control::

              guard.add_entities({
                  "CREDIT_CARD":      "hash",
                  "SPECIAL_CATEGORY": {"action": "redact", "layers": ["llm"]},
              })

        :attr:`Entity.ALL` may appear as an entry to expand to every known
        entity. The top-level ``action``/``layers`` act as defaults for any entry
        that does not specify its own. Any entity left without an action defaults
        to ``"hash"`` and logs a single warning.

        :raises ConfigError: if ``entities`` is not a mapping or iterable, or any
                            spec/action/entity is invalid.
        """
        if isinstance(entities, dict):
            specs = entities
        elif isinstance(entities, str | Entity):
            # A bare string/Entity is a common mistake — it would iterate chars.
            raise ConfigError(
                f"add_entities() expects a mapping or an iterable of entity types, "
                f"not a single {type(entities).__name__}. Use add_entity() for one entity."
            )
        else:
            try:
                specs = dict.fromkeys(entities, {})
            except TypeError as exc:
                raise ConfigError(
                    f"add_entities() expects a mapping or an iterable of entity types, "
                    f"got {type(entities).__name__}."
                ) from exc

        defaulted = False
        for name, spec in specs.items():
            if isinstance(spec, str):
                spec = {"action": spec}
            elif spec is None:
                spec = {}
            elif not isinstance(spec, dict):
                raise ConfigError(
                    f"Invalid spec for {name!r}: expected str, dict, or None, "
                    f"got {type(spec).__name__}."
                )
            entity_action = spec.get("action", action)
            if entity_action is None:
                entity_action = Action.HASH.value
                defaulted = True
            entity_layers = spec.get("layers", layers)
            names = sorted(KNOWN_ENTITY_TYPES) if self._is_all(name) else [name]
            for n in names:
                self._set_entity(n, enabled=True, action=entity_action, layers=entity_layers)
        if defaulted:
            logger.warning(
                "add_entities(): one or more entities were enabled without an action — "
                "defaulting to 'hash'. Pass action=... (or a per-entity action) to silence this."
            )
        self._rebuild()
        return self

    def remove_entity(self, entity_type: str | Entity) -> AIGuard:
        """
        Disable a single entity type (or every type via :attr:`Entity.All`).

        Pairs with :meth:`add_entity` — a common pattern is to enable everything
        and prune what you do not need::

            guard.add_entity(Entity.ALL, action="hash").remove_entity(Entity.ORG)

        Removing an entity that was never enabled is a no-op. An unknown entity
        *name* (likely a typo) logs a warning, like :meth:`add_entity`. Supports
        chaining.

        :raises ConfigError: if ``entity_type`` is not a ``str`` or ``Entity``.
        """
        if self._is_all(entity_type):
            self._disable_all()
        else:
            name = self._normalize_entity(entity_type)
            self._warn_if_unknown(name)
            self._disable_entity(name)
        self._rebuild()
        return self

    def remove_entities(self, entities) -> AIGuard:
        """
        Disable many entity types at once (single rebuild). Supports chaining.

        ``entities`` is an iterable of entity names/constants (``Entity.ALL`` is
        accepted and expands to every enabled entity).

        :raises ConfigError: if ``entities`` is not an iterable of entity types.
        """
        if isinstance(entities, str | Entity):
            raise ConfigError(
                f"remove_entities() expects an iterable of entity types, not a single "
                f"{type(entities).__name__}. Use remove_entity() for one entity."
            )
        try:
            items = list(entities)
        except TypeError as exc:
            raise ConfigError(
                f"remove_entities() expects an iterable of entity types, "
                f"got {type(entities).__name__}."
            ) from exc

        for entity_type in items:
            if self._is_all(entity_type):
                self._disable_all()
            else:
                name = self._normalize_entity(entity_type)
                self._warn_if_unknown(name)
                self._disable_entity(name)
        self._rebuild()
        return self

    def change_entity_action(self, entity_type: str | Entity, action: str | Action) -> AIGuard:
        """
        Change the action of an entity that is **currently enabled**.

        Unlike :meth:`add_entity`, this never enables a new entity: it only
        retargets the action (``warn`` / ``hash`` / ``redact`` / ``mask``) of an
        entity that is already active. If the entity was removed or was never
        added, it raises — enable it first with :meth:`add_entity`. The layers it
        runs on are left unchanged. Supports method chaining.

        :attr:`Entity.ALL` changes the action of *every* currently-enabled entity.

        :raises ConfigError: if ``entity_type`` is not a ``str``/``Entity``, the
                            ``action`` is invalid, or the entity (or, for
                            ``Entity.ALL``, *any* entity) is not currently enabled.
        """
        action = self._normalize_action(action)

        if self._is_all(entity_type):
            active = self._active_entities()
            if not active:
                raise ConfigError(
                    "change_entity_action(Entity.ALL) failed: no entities are currently "
                    "enabled. Enable some first with add_entity()."
                )
            for name in sorted(active):
                self._apply_action(name, action)
        else:
            name = self._normalize_entity(entity_type)
            if not self._is_entity_active(name):
                raise ConfigError(
                    f"Cannot change action for {name!r}: it is not enabled "
                    "(it was removed or never added). Enable it first with add_entity()."
                )
            self._apply_action(name, action)
        self._rebuild()
        return self

    # ------------------------------------------------------------------
    # Introspection (read the current policy)
    # ------------------------------------------------------------------

    def enabled_entities(self) -> set[str]:
        """Return the set of entity types currently enabled on any detector layer.

        ::

            guard.add_entity(Entity.ALL, action="hash").remove_entity(Entity.ORG)
            "ORG" in guard.enabled_entities()      # False
        """
        return self._active_entities()

    def get_entity_action(self, entity_type: str | Entity) -> str | None:
        """Return the action of a currently-enabled entity, or ``None`` if it is
        not enabled.

        ::

            guard.get_entity_action(Entity.EMAIL)   # "hash" | "warn" | … | None

        :raises ConfigError: if ``entity_type`` is not a ``str``/``Entity``, or is
                            :attr:`Entity.ALL` (use :meth:`entity_policy` instead).
        """
        if self._is_all(entity_type):
            raise ConfigError(
                "get_entity_action() does not accept Entity.ALL — use entity_policy()."
            )
        name = self._normalize_entity(entity_type)
        if not self._is_entity_active(name):
            return None
        return self._entity_action(name)

    def entity_policy(self) -> dict[str, str]:
        """Return a ``{entity_type: action}`` mapping of every enabled entity.

        ::

            guard.entity_policy()   # {"CREDIT_CARD": "hash", "EMAIL": "warn", …}
        """
        policy: dict[str, str] = {}
        for name in sorted(self._active_entities()):
            action = self._entity_action(name)
            if action is not None:
                policy[name] = action
        return policy

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
    ) -> AIGuard:
        """
        Enable the SpaCy NER layer with an explicit model. Supports chaining.

        Mirrors :meth:`with_llm`. Pass ``language=`` (recommended; a list enables
        multilingual NER) or ``spacy_model=`` (explicit package name(s)).

        ::

            guard = AIGuard(salt="s").with_ner(language=Language.EN)
            guard = AIGuard(salt="s").with_ner(spacy_model=["en_core_web_sm", "de_core_news_sm"])

        :raises ConfigError: if neither ``language`` nor ``spacy_model`` is given.
        """
        if language is None and spacy_model is None:
            raise ConfigError(
                "with_ner() requires a model — pass language=... (e.g. Language.EN) "
                "or spacy_model=...; ai-guard ships no default model."
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
    ) -> AIGuard:
        """
        Enable the on-prem LLM detector. Supports chaining; mirrors :meth:`with_ner`.

        A fluent alternative to the constructor's ``llm_*`` arguments — keeps the
        LLM configuration in one place::

            guard = (
                AIGuard(salt="s")
                .with_ner(language=Language.TR)
                .with_llm(backend=Backend.OLLAMA, model="llama3.2", adjudicate=True)
            )

        ``backend`` is the backend *type* (:class:`~ai_guard.Backend`); the
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

    # ------------------------------------------------------------------
    # Internal entity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_all(entity_type: object) -> bool:
        """True if *entity_type* is the :attr:`Entity.ALL` sentinel.

        (``Entity.All`` is an alias of ``Entity.ALL``, so it is covered too.)
        """
        return entity_type is Entity.ALL or entity_type == Entity.ALL.value

    @staticmethod
    def _normalize_entity(entity_type: str | Entity) -> str:
        """Coerce an Entity/str to its canonical string value, validating the type.

        Note: ``str(Entity.EMAIL) == "Entity.EMAIL"``, so we read ``.value`` rather
        than calling ``str()``.
        """
        if isinstance(entity_type, Entity):
            return entity_type.value
        if isinstance(entity_type, str):
            return entity_type
        raise ConfigError(f"entity_type must be a str or Entity, got {type(entity_type).__name__}.")

    @staticmethod
    def _action_or_default(action: str | Action | None) -> str | Action:
        """Return *action*, or the default ``hash`` (with a warning) when unspecified."""
        if action is None:
            logger.warning(
                "No action specified — defaulting to 'hash' (the safest action). "
                "Pass action=... (e.g. Action.WARN) to choose explicitly and silence this warning."
            )
            return Action.HASH.value
        return action

    @staticmethod
    def _resolve_language_models(
        language: str | Language | list[str | Language], spacy_size: str
    ) -> list[str]:
        """Resolve one or more languages (+ size tier) to concrete SpaCy model names."""
        from ai_guard.ner.spacy_catalog import (
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
                    f"Unsupported language {code!r}. Supported: {supported}. "
                    "See: python -m ai_guard spacy list"
                )
            models.append(info.name)
        return list(dict.fromkeys(models))  # dedupe, preserve order

    @staticmethod
    def _normalize_action(action: str | Action) -> str:
        """Validate and coerce an Action/str to its canonical string value."""
        # ValueError: unknown string value; TypeError: unhashable type (e.g. list).
        try:
            return Action(action).value
        except (ValueError, TypeError):
            raise ConfigError(
                f"Invalid action {action!r}. Valid values: 'warn', 'hash', 'mask', 'redact'"
            ) from None

    def _warn_if_unknown(self, entity_type: str) -> None:
        """Log a warning if *entity_type* is neither a known nor a custom entity."""
        if entity_type not in self._config.get("custom_patterns", {}):
            warn_unknown_entity(entity_type)

    def _is_entity_active(self, entity_type: str) -> bool:
        """True if *entity_type* is currently enabled on any *active* detector layer."""
        ent = self._config.get("entities", {}).get(entity_type)
        if ent and ent.get("enabled"):
            return True
        # LLM entities only count when the LLM layer itself is enabled.
        llm_cfg = self._config.get("llm_detector", {})
        if llm_cfg.get("enabled"):
            llm = llm_cfg.get("entities", {}).get(entity_type)
            if llm and llm.get("enabled"):
                return True
        return False

    def _entity_action(self, entity_type: str) -> str | None:
        """Return the configured action for *entity_type*, or None if unset."""
        ent = self._config.get("entities", {}).get(entity_type)
        if ent and "action" in ent:
            return str(ent["action"])
        llm = self._config.get("llm_detector", {}).get("entities", {}).get(entity_type)
        if llm and "action" in llm:
            return str(llm["action"])
        return None

    def _active_entities(self) -> set[str]:
        """The set of entity types currently enabled on any layer."""
        names = set(self._config.get("entities", {})) | set(
            self._config.get("llm_detector", {}).get("entities", {})
        )
        return {name for name in names if self._is_entity_active(name)}

    def _apply_action(self, entity_type: str, action: str) -> None:
        """Update an entity's action on every layer where it exists (no rebuild)."""
        entities = self._config.get("entities", {})
        if entity_type in entities:
            entities[entity_type]["action"] = action
        llm_entities = self._config.get("llm_detector", {}).get("entities", {})
        if entity_type in llm_entities:
            llm_entities[entity_type]["action"] = action

    def _disable_entity(self, entity_type: str) -> None:
        """Set an entity's ``enabled`` flag to False on every layer (no rebuild)."""
        entities = self._config.get("entities", {})
        if entity_type in entities:
            entities[entity_type]["enabled"] = False
        llm_entities = self._config.get("llm_detector", {}).get("entities", {})
        if entity_type in llm_entities:
            llm_entities[entity_type]["enabled"] = False

    def _disable_all(self) -> None:
        """Disable every configured entity on every layer (no rebuild)."""
        for name in list(self._config.get("entities", {})):
            self._disable_entity(name)
        for name in list(self._config.get("llm_detector", {}).get("entities", {})):
            self._disable_entity(name)

    def _set_entity(
        self,
        entity_type: str | Entity,
        *,
        enabled: bool,
        action: str | Action,
        layers: list[str] | None,
    ) -> None:
        """Mutate config for one entity across the chosen layers (no rebuild)."""
        # Normalize the entity type to its canonical string, validating the
        # argument type. (Entity.ALL must be expanded by the caller, never reach here.)
        if self._is_all(entity_type):
            raise ConfigError(
                "Entity.ALL cannot be set directly — it is expanded by add_entity()/"
                "add_entities() into the individual known entity types."
            )
        entity_type = self._normalize_entity(entity_type)
        action = self._normalize_action(action)
        self._warn_if_unknown(entity_type)

        if layers is None:
            target = [lyr for lyr, ents in _LAYER_ENTITIES.items() if entity_type in ents]
            if not target:  # unknown / custom entity → default to regex
                target = ["regex"]
        else:
            invalid = set(layers) - _VALID_LAYERS
            if invalid:
                raise ConfigError(
                    f"Invalid layer(s) {sorted(invalid)}. Valid: {sorted(_VALID_LAYERS)}"
                )
            target = list(layers)

        # config["entities"] holds the action (always, so the engine can apply it)
        # and the regex/NER enabled flag.
        non_llm = ("regex" in target) or ("ner" in target)
        self._config.setdefault("entities", {})[entity_type] = {
            "enabled": enabled and non_llm,
            "action": action,
        }
        # The LLM layer keeps its own enabled set.
        if "llm" in target:
            llm_entities = self._config.setdefault("llm_detector", {}).setdefault("entities", {})
            llm_entities[entity_type] = {"enabled": enabled, "action": action}

    def set_salt(self, salt: str) -> AIGuard:
        """Update the hash salt."""
        self._config["salt"] = salt
        self._rebuild()
        return self

    def add_allowlist(self, values: list[str]) -> AIGuard:
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

    def add_denylist(self, entries: list[dict[str, str]]) -> AIGuard:
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
                "the AIGUARD_SALT environment variable in production."
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
        enabled_regex = {e for e in _REGEX_ENTITIES if entity_cfg.get(e, {}).get("enabled", False)}
        if enabled_regex or custom_patterns:
            self._detectors.append(RegexDetector(enabled_regex, custom_patterns=custom_patterns))

        # SpaCy NER detector(s) (optional). A list of models loads one detector
        # per language (multilingual NER); the engine merges their spans.
        if self._config.get("use_ner", True):
            enabled_ner = {e for e in _NER_ENTITIES if entity_cfg.get(e, {}).get("enabled", False)}
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
                        from ai_guard.detectors.ner_detector import NERDetector

                        if auto_download:
                            from ai_guard.ner.downloader import ensure_model

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

        # LLM detector (optional)
        llm_cfg = self._config.get("llm_detector", {})
        if llm_cfg.get("enabled", False):
            self._detectors.append(self._build_llm_detector(llm_cfg))

        self._engine = DetectionEngine(self._config, self._detectors)

    def _build_llm_detector(self, llm_cfg: dict[str, Any]) -> BaseDetector:
        """Build the LLM detector according to configuration."""
        from ai_guard.detectors.llm_detector import LLMDetector
        from ai_guard.llm.backends.base import BaseLLMBackend

        backend: BaseLLMBackend
        backend_name = llm_cfg.get("backend", "ollama")
        model = llm_cfg.get("model", "llama3.2")
        base_url = llm_cfg.get("base_url", "http://localhost:11434")
        api_key = llm_cfg.get("api_key", "")
        timeout = llm_cfg.get("timeout", 60)
        allow_http = llm_cfg.get("allow_http", False)

        if backend_name == "ollama":
            from ai_guard.llm.backends.ollama import OllamaBackend
            from ai_guard.llm.model_manager import ModelManager

            ollama_backend = OllamaBackend(base_url=base_url, model=model, allow_http=allow_http)
            if llm_cfg.get("auto_pull", False):
                mgr = ModelManager(ollama_backend)
                mgr.ensure_available(model, verbose=True)
            backend = ollama_backend
        elif backend_name == "openai_compatible":
            from ai_guard.llm.backends.openai_compat import OpenAICompatBackend

            backend = OpenAICompatBackend(
                base_url=base_url, model=model, api_key=api_key, allow_http=allow_http
            )
        elif backend_name == "transformers":
            from ai_guard.llm.backends.transformers_backend import TransformersBackend

            backend = TransformersBackend(
                model=model,
                device_map=llm_cfg.get("device_map", "auto"),
                load_in_8bit=llm_cfg.get("load_in_8bit", False),
                load_in_4bit=llm_cfg.get("load_in_4bit", False),
            )
        else:
            raise ConfigError(
                f"Unknown LLM backend: {backend_name!r}. "
                "Valid values: 'ollama', 'openai_compatible', 'transformers'"
            )

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

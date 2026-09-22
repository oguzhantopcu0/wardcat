from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from wardcat.detectors.llm_detector import LLMDetector

from wardcat._entity_policy import EntityPolicyMixin
from wardcat.config.loader import _validate_denylist, load_config
from wardcat.core.engine import DetectionEngine
from wardcat.core.models import KNOWN_ENTITY_TYPES, Layer, ScanResult, SensitivityVerdict
from wardcat.core.registry import (
    LAYER_ENTITIES,
    NER_ENTITIES,
    REGEX_ENTITIES,
    VALID_LAYERS,
)
from wardcat.detectors.base import BaseDetector
from wardcat.detectors.regex_detector import RegexDetector
from wardcat.exceptions import ConfigError, DegradedScanError, UnsupportedLanguageError
from wardcat.llm.backends.base import Backend
from wardcat.llm.prompt import build_classification_messages, parse_classification
from wardcat.ner.spacy_catalog import Language
from wardcat.utils.text import chunk_by_paragraph

logger = logging.getLogger(__name__)

# Max characters per LLM call in is_sensitive(). Classification tolerates more
# context than span extraction, but long inputs are still chunked (any chunk
# sensitive → the whole text is sensitive) so nothing is silently truncated.
_SENSITIVITY_CHUNK_CHARS = 4000
_DEFAULT_MAX_TEXT_BYTES = 500_000


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


def _merge_verdicts(verdicts: list[SensitivityVerdict]) -> SensitivityVerdict:
    """One verdict for a text judged in chunks: sensitive if any chunk is."""
    sensitive = [v for v in verdicts if v.sensitive]
    if not sensitive:
        return SensitivityVerdict(False)
    categories = tuple(dict.fromkeys(c for v in sensitive for c in v.categories))
    return SensitivityVerdict(True, categories, sensitive[0].reason)


def _ner_fallback_warning(requested: str, used: str) -> str:
    """Describe a SpaCy model substitution, calling out a change of language."""
    message = (
        f"SpaCy model {requested!r} is not installed; the NER layer is using {used!r} instead."
    )
    wanted_lang, used_lang = requested.split("_")[0], used.split("_")[0]
    if wanted_lang != used_lang:
        message += (
            f" That model is for a different language ({used_lang!r}, not {wanted_lang!r}), "
            "so most names in the requested language will be missed."
        )
    return message + f" Install the requested model with: python -m spacy download {requested}"


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

    Configuration is explicit. The constructor takes only ``salt`` and an optional
    YAML ``config_path``; every detection layer is configured with a fluent
    builder — :meth:`with_ner` and :meth:`with_llm` — or in the YAML file. The
    library does **not** read environment variables: read any secrets in your own
    application and hand them to the constructor. Builders are chainable and their
    order does not matter — the final configuration is what counts::

        from wardcat import Wardcat, Language

        # LLM layer
        guard = Wardcat(salt="s").with_llm(model="llama3.1:8b")

        # NER layer — needs an explicit model (wardcat ships no default). Choose
        # one via language= (recommended) or spacy_model=:
        guard = Wardcat(salt="s").with_ner(language=Language.DE, spacy_size="md")
        guard = Wardcat(salt="s").with_ner(language=[Language.DE, Language.FR])
        guard = Wardcat(salt="s").with_ner(spacy_model=["en_core_web_sm", "de_core_news_sm"])
        # Supported languages: en, de, fr, es, it, nl, pt, tr; sizes sm/md/lg/trf.
        # A named-but-missing model is auto-downloaded (auto_download=False to disable).
        # For mixed-language text without extra models, use the LLM layer, whose
        # prompt is multilingual.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        salt: str = "",
    ) -> None:
        self._config = load_config(config_path)

        # Constructor arguments override YAML
        if salt:
            self._config["salt"] = salt

        # Detection layers are configured with the fluent builders — with_ner()
        # and with_llm() — or a YAML config_path, never constructor arguments.
        # Ensure the LLM sub-config exists for the YAML/builder path.
        self._config.setdefault("llm_detector", {})

        # A YAML config may still switch NER on; it must then name a model, since
        # wardcat ships no default. (The builder path always sets one.)
        if self._config.get("use_ner") and not (
            self._config.get("spacy_models") or self._config.get("spacy_model")
        ):
            raise ConfigError(
                "use_ner is on but no SpaCy model was given. Set spacy_model in the "
                "YAML config, or configure NER with with_ner(language=...) / "
                "with_ner(spacy_model=...); wardcat ships no default model."
            )

        # Entities whose enabled layer is missing — warned about once, lazily, at
        # scan time (see _warn_orphan_entities). Chains configure layers and
        # entities in any order, so an init/rebuild-time check would false-fire.
        self._orphan_warned: set[str] = set()

        # Entity types this caller configured by hand (add_entity / add_entities /
        # change_entity_action). The LLM layer ships its own default entity policy,
        # so this is the only way to tell "the user asked for PERSON" apart from
        # "with_llm() switched PERSON on" — see _warn_implicit_llm_entities.
        self._explicit_entities: set[str] = set()
        # A caller who passed a policy file chose every entity in it; nothing about
        # that configuration is implicit, so the warning is skipped for them.
        self._policy_from_file = config_path is not None
        self._implicit_llm_warned = False
        # Place names and group names moved out of ADDRESS/ORG into their own
        # types. A configuration written before that silently stops covering
        # them — see _warn_ner_type_split.
        self._ner_split_warned = False

        # Warn at most once when a hash action is active without a salt. Checked
        # in _rebuild() too, since entities are opt-in and usually added after init.
        self._salt_warned = False
        self._default_action_warned = False
        # A YAML `preset:` is the base; the file's own `entities:` win over it.
        preset_name = self._config.pop("preset", None)
        if preset_name:
            self._apply_preset(preset_name, keep_existing=True)
        self._rebuild()

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan(self, text: str) -> ScanResult:
        """Scan text and return a ScanResult."""
        self._warn_about_configuration()
        return self._engine.scan(text)

    async def scan_async(self, text: str) -> ScanResult:
        """Async scan — uses native async I/O for the LLM backend when available.

        CPU-bound detectors (regex, SpaCy NER) run in a thread pool;
        the LLM detector (if enabled) uses ``httpx.AsyncClient`` natively,
        so multiple concurrent calls do not block each other.
        """
        self._warn_about_configuration()
        return await self._engine.scan_async(text)

    # ------------------------------------------------------------------
    # Semantic sensitivity check (LLM-only)
    # ------------------------------------------------------------------

    def is_sensitive(self, text: str) -> bool:
        """Return whether *text* contains sensitive information, judged semantically.

        The boolean of :meth:`classify`: a general, holistic LLM decision — *not*
        the per-entity detection of :meth:`scan`. It asks the configured LLM
        whether the text as a whole contains sensitive information (PII,
        credentials, financial, health, or confidential business data). Useful as
        a lightweight guardrail before sending text to an external service.

        Requires the LLM layer (:meth:`with_llm`); no entities need to be enabled
        and no regex/NER runs. Empty text is ``False``. Fail-closed: if the LLM
        backend is unreachable the underlying error propagates, and an answer
        that cannot be read counts as sensitive, so a guardrail never silently
        treats sensitive text as safe.

        :raises ConfigError: if the LLM layer is not configured.
        """
        # Only the boolean is wanted, so the first sensitive chunk settles it.
        return self._classify(text, stop_at_first=True).sensitive

    async def is_sensitive_async(self, text: str) -> bool:
        """Async variant of :meth:`is_sensitive` (native async LLM I/O when available)."""
        return (await self._classify_async(text, stop_at_first=True)).sensitive

    def classify(self, text: str) -> SensitivityVerdict:
        """Decide whether *text* is sensitive, and of what kind.

        Like :meth:`is_sensitive`, one holistic LLM judgement over the whole
        text — but the answer names the categories present (``pii``,
        ``credentials``, ``financial``, ``health``, ``special_category``,
        ``business_confidential``) and carries the model's one-line reason, so a
        caller can route on the kind: block health data, allow business data
        inside the company, log the rest. Long texts are judged in chunks and the
        categories merged; the reason is the first sensitive chunk's.

        An answer that cannot be read is *sensitive* with the category
        ``"unknown"`` — the guardrail fails closed. ``reason`` may quote the
        text and is as sensitive as the input.

        :raises ConfigError: if the LLM layer is not configured.
        """
        return self._classify(text, stop_at_first=False)

    async def classify_async(self, text: str) -> SensitivityVerdict:
        """Async variant of :meth:`classify`."""
        return await self._classify_async(text, stop_at_first=False)

    def _classify(self, text: str, *, stop_at_first: bool) -> SensitivityVerdict:
        detector = self._require_llm("classify")
        self._check_text_size(text)
        if not text.strip():
            return SensitivityVerdict(False)
        language = self._config.get("llm_detector", {}).get("language")
        verdicts = []
        for chunk, _ in chunk_by_paragraph(text, _SENSITIVITY_CHUNK_CHARS):
            if not chunk.strip():
                continue
            reply = detector.complete_messages(build_classification_messages(chunk, language))
            verdicts.append(parse_classification(reply))
            if stop_at_first and verdicts[-1].sensitive:
                break
        return _merge_verdicts(verdicts)

    async def _classify_async(self, text: str, *, stop_at_first: bool) -> SensitivityVerdict:
        detector = self._require_llm("classify_async")
        self._check_text_size(text)
        if not text.strip():
            return SensitivityVerdict(False)
        language = self._config.get("llm_detector", {}).get("language")
        verdicts = []
        for chunk, _ in chunk_by_paragraph(text, _SENSITIVITY_CHUNK_CHARS):
            if not chunk.strip():
                continue
            reply = await detector.complete_messages_async(
                build_classification_messages(chunk, language)
            )
            verdicts.append(parse_classification(reply))
            if stop_at_first and verdicts[-1].sensitive:
                break
        return _merge_verdicts(verdicts)

    def _check_text_size(self, text: str) -> None:
        """Reject oversized input (mirrors the engine's DoS guard for scan())."""
        limit = self._config.get("max_text_bytes", _DEFAULT_MAX_TEXT_BYTES)
        byte_len = len(text.encode("utf-8", errors="replace"))
        if byte_len > limit:
            raise ValueError(
                f"Input text is too large: {byte_len:,} bytes (maximum: {limit:,} bytes). "
                "Split the text into smaller chunks."
            )

    def _require_llm(self, feature: str) -> LLMDetector:
        """Return the configured LLM detector or raise if none is set."""
        detector = self._llm_detector
        if detector is None:
            raise ConfigError(
                f"{feature}() needs the LLM layer — configure it with "
                "with_llm(...) (e.g. Wardcat().with_llm(model='llama3.1:8b'))."
            )
        return detector

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

        # Workers call the engine directly, so the one-time configuration warnings
        # that scan() logs have to be raised here, once for the whole batch.
        self._warn_about_configuration()
        workers = max_workers or self._config.get("scan_batch_workers", 4)

        results: list[ScanResult | None] = [None] * len(texts)

        def _scan_one(idx: int, text: str) -> tuple[int, ScanResult]:
            try:
                return idx, self._engine.scan(text)
            # Strict mode means no degraded result may come back, from a batch
            # either; the item is not turned into a scan_error entry.
            except DegradedScanError:
                raise
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
                    warnings=list(self._engine.build_warnings),
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
            except DegradedScanError:
                raise
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
                    warnings=list(self._engine.build_warnings),
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
    def supported_entities(layer: str | Layer | None = None) -> frozenset[str]:
        """Return the entity types wardcat can detect (discoverability helper).

        ::

            Wardcat.supported_entities()            # every known entity type
            Wardcat.supported_entities("regex")     # only what the regex layer detects
            Wardcat.supported_entities("ner")       # PERSON, ORG, ADDRESS
            Wardcat.supported_entities("llm")       # contextual/semantic types

        :param layer: ``None`` → all known types; or one of ``"regex"``,
                      ``"ner"``, ``"llm"`` for that layer's set.
        :raises ConfigError: if ``layer`` is not a known layer.
        """
        if layer is None:
            return frozenset(KNOWN_ENTITY_TYPES)
        if isinstance(layer, Layer):
            layer = layer.value
        if layer not in LAYER_ENTITIES:
            raise ConfigError(f"Unknown layer {layer!r}. Valid layers: {sorted(VALID_LAYERS)}")
        return LAYER_ENTITIES[layer]

    # ------------------------------------------------------------------
    # Layer builders (the only way to enable the NER/LLM layers programmatically)
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
        base_url: str | None = None,
        api_key: str = "",
        timeout: int = 60,
        allow_http: bool = False,
        adjudicate: bool = False,
        auto_pull: bool = False,
        device_map: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        dtype: str | None = None,
        language: str | Language | None = None,
        circuit_failures: int = 3,
        circuit_cooldown: float = 30.0,
    ) -> Wardcat:
        """
        Enable the on-prem LLM detector. Supports chaining, like :meth:`with_ner`.

        .. warning::
            Unlike ``with_ner()``, this does **not** leave detection fully opt-in.
            The LLM layer carries its own default entity policy, so ``with_llm()``
            switches on around fifteen entity types with the actions that policy
            names (``PERSON`` → ``hash``, ``EMAIL`` → ``warn``, …) — not the action
            you pass to :meth:`add_entity` for something else. They are listed in a
            one-time warning at the first scan. Override one with
            ``add_entity(name, action, layers=["llm"])`` or drop it with
            ``remove_entity(name)``; pass a YAML ``config_path`` to replace the
            policy wholesale.

        The fluent way to configure the LLM layer (the constructor takes only
        ``config_path`` and ``salt``) — keeps the LLM configuration in one place::

            guard = (
                Wardcat(salt="s")
                .with_ner(language=Language.TR)
                .with_llm(backend=Backend.OLLAMA, model="llama3.2", adjudicate=True)
            )

        ``backend`` is the backend *type* (:class:`~wardcat.Backend`); the
        *address* goes to ``base_url``.

        :param base_url: the backend's address. Leave it unset to use the
            backend's own default — ``http://localhost:11434`` for ``ollama``
            and ``openai_compatible``, ``http://localhost:8000/v1`` for
            ``vllm``. Only pass it to point at a non-default host/port; passing
            it here would otherwise override the backend-specific default (so
            selecting ``vllm`` without a ``base_url`` must still reach vLLM,
            not Ollama).
        :param dtype: weight dtype for the ``transformers`` backend, as a torch
            dtype name (``"float16"``, ``"bfloat16"``, ``"float32"``); an
            unknown name raises. Left unset the default is ``bfloat16``, except
            on a pre-Ampere CUDA card, which has no bf16 support and gets
            ``float16``. On Apple Silicon ``bfloat16`` is emulated and fp16
            *ought* to be faster, but loading as ``float16`` with
            ``device_map="auto"`` on MPS segfaults on the supported
            torch/transformers versions — hence the argument rather than a
            different default. Ignored by the other backends, which do not load
            weights themselves.
        :param language: selects a localized system prompt for :meth:`is_sensitive`
            (``tr``/``de``/``fr``; anything else uses the English, multilingual-aware
            prompt). It does not change the entity-detection prompt used by
            :meth:`scan`, which is multilingual by design.
        :param circuit_failures: consecutive backend failures after which the
            LLM layer is skipped without being called, so an outage does not make
            every scan wait the full ``timeout``. ``0`` disables the breaker.
        :param circuit_cooldown: seconds the layer stays skipped before one call
            is tried again. While skipped, each scan carries a warning naming
            the open circuit; :meth:`is_sensitive` raises instead.
        """
        lang_code = language.value if isinstance(language, Language) else language
        llm_cfg = self._config.setdefault("llm_detector", {})
        llm_cfg.update(
            {
                "enabled": True,
                "backend": backend.value if isinstance(backend, Backend) else backend,
                "model": model,
                "api_key": api_key,
                "timeout": timeout,
                "allow_http": allow_http,
                "adjudicate": adjudicate,
                "auto_pull": auto_pull,
                "device_map": device_map,
                "load_in_8bit": load_in_8bit,
                "load_in_4bit": load_in_4bit,
                "dtype": dtype,
                "language": lang_code,
                "circuit_failures": circuit_failures,
                "circuit_cooldown": circuit_cooldown,
            }
        )
        # Only pin base_url when the caller gave one; otherwise leave it out so
        # each backend factory applies its own default (Ollama 11434 vs vLLM
        # 8000/v1). A prior with_llm() call's base_url is cleared here too.
        if base_url is not None:
            llm_cfg["base_url"] = base_url
        else:
            llm_cfg.pop("base_url", None)
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
            get_models_by_language,
            resolve_model,
            supported_languages,
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
                raise UnsupportedLanguageError(
                    f"Unsupported language {code!r}. Supported: {supported_languages()}."
                )
            models.append(info.name)
        return list(dict.fromkeys(models))  # dedupe, preserve order

    def with_phone_regions(self, *regions: str) -> Wardcat:
        """Detect national phone formats for *regions* via libphonenumber.

        The built-in ``PHONE`` pattern is precision-first and covers Turkish,
        French and German national formats plus E.164 — a number written the way
        it is written in Manchester or Madrid falls through it. Naming the regions
        you actually serve swaps in libphonenumber for those formats::

            guard = Wardcat(salt=s).add_entity(Entity.PHONE).with_phone_regions("GB", "ES")

        Regions are CLDR two-letter codes. Needs the extra: ``pip install
        'wardcat[phone]'`` — without it the built-in pattern is used and a warning
        is logged. Call with no arguments to go back to the pattern.

        Matches are reported at 0.90 confidence, not the 0.97 of the built-in
        pattern: libphonenumber validates against each region's numbering plan,
        which is far stronger than a bare digit run but weaker than a checksum, and
        every extra region widens what counts as a number. Add the regions you
        serve, not every region there is.
        """
        codes = [r.strip().upper() for r in regions if r and r.strip()]
        self._config["phone_regions"] = codes
        self._rebuild()
        return self

    def with_min_confidence(self, minimum: float) -> Wardcat:
        """Set the confidence floor: spans scoring below *minimum* are dropped.

        Every detection carries a confidence, tiered by how strong the evidence
        is — a checksum-verified card is 1.0, a distinctive format such as an
        email is 0.97, a model layer is 0.85, a keyword-heuristic address is
        0.90, and a checksum whose own odds are weak (the ABA mod-10, the NHS
        mod-11, the IMEI Luhn) with no supporting keyword nearby is 0.70.

        The default floor is ``0.8``, which sits between that last tier and
        everything else: those uncued matches are found but not acted on. Lower
        it to trade precision for recall::

            guard.with_min_confidence(0.6)   # act on uncued checksum matches too

        Set it to ``0`` to act on everything a layer reports.

        :raises ConfigError: if *minimum* is not a number between 0 and 1.
        """
        from wardcat.config.loader import _validate_min_confidence

        _validate_min_confidence(minimum)
        self._config["min_confidence"] = float(minimum)
        self._rebuild()
        return self

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

        Each entry has an ``entity_type`` and either an exact ``value`` or a regex
        ``pattern``. The action applied is taken from the entity's config (same as
        regular detections).  Supports method chaining::

            guard.add_denylist([
                {"value": "John Smith",    "entity_type": "PERSON"},
                {"pattern": r"\\bPRJ-\\d{4}\\b", "entity_type": "CUSTOM_SECRET"},
            ])

        Entries are validated exactly as a YAML ``denylist`` is, and all of them
        before any is added: a pattern that is not valid regex, or that backtracks
        catastrophically, is refused — a match cannot be interrupted once a scan
        has started it.

        :param entries: List of dicts with ``entity_type`` and ``value`` or ``pattern``.
        :raises ConfigError: if any entry is invalid.
        """
        _validate_denylist(entries)
        self._config.setdefault("denylist", []).extend(entries)
        self._rebuild()
        return self

    @staticmethod
    def supported_presets() -> tuple[str, ...]:
        """The names :meth:`with_preset` accepts."""
        from wardcat.presets import supported_presets

        return supported_presets()

    def with_preset(self, name: str) -> Wardcat:
        """Enable the entities of a starting policy, with the actions it names.

        A preset is an entity → action mapping modelled on a data-protection
        regime (``"kvkk"``, ``"gdpr"``, ``"pci_dss"``, ``"hipaa_lite"``,
        ``"secrets_only"``). It switches no layer on: names need
        :meth:`with_ner` or :meth:`with_llm`, special-category data needs
        :meth:`with_llm`, and an entity left without its layer is reported as
        uncovered at the first scan, as always. Adjust afterwards with
        :meth:`change_entity_action` and :meth:`remove_entity`. YAML: ``preset: kvkk``,
        with the file's own ``entities`` taking precedence.

        See the presets guide for what each one enables and, as importantly,
        what it does not cover.

        :raises ConfigError: for an unknown preset name.
        """
        self._apply_preset(name, keep_existing=False)
        self._rebuild()
        return self

    def _apply_preset(self, name: str, *, keep_existing: bool) -> None:
        from wardcat.presets import get_preset

        preset = get_preset(name)
        configured = self._config.get("entities", {})
        for entity, action in preset.entities.items():
            if keep_existing and entity in configured:
                continue
            self._set_entity(entity, enabled=True, action=action, layers=None)

    def with_strict(self, enabled: bool = True) -> Wardcat:
        """Refuse any scan that covers less than was configured.

        By default a layer that cannot run — a SpaCy model that failed to load,
        an LLM backend that is down — is reported on :attr:`ScanResult.warnings`
        and the partial result is returned, which is right for a chat guard
        that must answer. A pipeline that indexes documents wants the opposite:
        better no result than a document stored with names in it. With strict
        on, such a condition raises :class:`~wardcat.DegradedScanError` — at
        build time for problems known then, at scan time for the rest, and
        from :meth:`scan_batch` too, which otherwise turns errors into
        ``scan_error`` entries.

        YAML: ``strict: true``.
        """
        self._config["strict"] = bool(enabled)
        self._rebuild()
        return self

    def with_propagation(self, *, enabled: bool = True, min_length: int = 3) -> Wardcat:
        """Redact **every** occurrence of a value once any layer detects it.

        Model-based layers (NER/LLM) often report a repeated value only
        once, which would leave the other occurrences unredacted. With
        propagation on, a value detected anywhere is anonymized at every
        whole-token occurrence in the text — using that value's entity type and
        action. Deterministic regex spans still win overlaps, so a propagated
        match never displaces a checksum-validated one. Chainable::

            guard = Wardcat(salt="s").with_ner().add_entity("PERSON").with_propagation()

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

    def _warn_about_configuration(self) -> None:
        """Log the once-only configuration warnings; run before any scan starts."""
        self._warn_orphan_entities()
        self._warn_implicit_llm_entities()
        self._warn_ner_type_split()

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
                "same hash, leaving them open to rainbow-table attacks). Pass salt=... to "
                "Wardcat(...) in production — the library never reads environment variables."
            )

    def _warn_orphan_entities(self) -> None:
        """Warn (once each) about entities enabled with no active layer to detect them.

        Enabling an entity whose supporting layer is off is a silent no-op — e.g.
        ``add_entity(Entity.PERSON)`` (a NER/LLM type) with neither ``with_ner()``
        nor ``with_llm()`` configured. Checked lazily at scan time (not during a
        build) so a builder chain can configure entities and layers in any order
        without false warnings.

        Intent is read from the shared entity map (what ``add_entity`` turns on);
        the LLM layer's default entity map is not a user signal, so it is not
        consulted here.
        """
        intended = {e for e, cfg in self._config.get("entities", {}).items() if cfg.get("enabled")}

        covered: set[str] = set()
        for detector in self._detectors:
            covered |= getattr(detector, "enabled_entities", set())

        orphans = intended - covered - self._orphan_warned
        if not orphans:
            return
        self._orphan_warned |= orphans
        for entity in sorted(orphans):
            hint = (
                "call with_llm()"
                if entity not in REGEX_ENTITIES and entity not in NER_ENTITIES
                else "call with_ner() and/or with_llm()"
            )
            logger.warning(
                "Entity %r is enabled but no active layer detects it — it will never "
                "be flagged. Enable a layer that supports it (%s), or target a layer "
                "explicitly with add_entity(%r, layers=[...]).",
                entity,
                hint,
                entity,
            )

    def _warn_implicit_llm_entities(self) -> None:
        """Warn once when the LLM layer detects entities the caller never configured.

        ``with_ner()`` enables no entity on its own; ``with_llm()`` does, because
        the LLM layer carries its own default entity policy (see
        ``DEFAULT_CONFIG["llm_detector"]["entities"]``). So a chain like::

            Wardcat(salt=s).with_llm(...).add_entity(Entity.EMAIL, Action.TOKENIZE)

        detects and anonymizes a dozen more types than the one that was asked for,
        under the LLM policy's own actions rather than the one just configured.
        That is long-standing behaviour and stays — but it is surprising enough to
        say out loud once, lazily at scan time so a builder chain can configure
        entities and layers in any order.
        """
        if self._implicit_llm_warned or self._policy_from_file:
            return
        llm_cfg = self._config.get("llm_detector", {})
        # The default entity map is present even with the layer off; nothing is
        # detected then, so there is nothing to point out.
        if not llm_cfg.get("enabled", False):
            return
        llm_entities = llm_cfg.get("entities", {})
        implicit = {
            name: cfg.get("action", "warn")
            for name, cfg in llm_entities.items()
            if cfg.get("enabled") and name not in self._explicit_entities
        }
        if not implicit:
            return
        self._implicit_llm_warned = True
        logger.warning(
            "with_llm() also switched on the LLM layer's own default entity policy: "
            "%s — these are detected and anonymized under those actions even though "
            "you did not configure them (unlike with_ner(), which enables nothing on "
            "its own). Take control of one with add_entity(name, action, "
            "layers=['llm']), or switch it off with remove_entity(name).",
            ", ".join(f"{name} ({action})" for name, action in sorted(implicit.items())),
        )

    # Spans the NER layer used to report under another type, and the type each
    # one moved to. Both moves are corrections — a city is not a street address,
    # and a nationality is not a company — but a configuration written before
    # them keeps working and quietly covers less.
    _NER_TYPE_SPLIT: dict[str, tuple[str, str]] = {
        "ADDRESS": ("LOCATION", "countries, cities and regions (SpaCy GPE/LOC)"),
        "ORG": ("NRP", "nationality, religious and political groups (SpaCy NORP)"),
    }

    def _warn_ner_type_split(self) -> None:
        """Warn once when a moved span type is enabled but its new home is not.

        Nothing breaks and nothing is over-detected; the spans simply stop being
        reported, which is the failure that does not announce itself. Said once,
        lazily at scan time, like the other two warnings here.
        """
        if self._ner_split_warned or self._policy_from_file:
            return
        if not self._config.get("use_ner", False):
            return
        enabled = self._active_entities()
        moved = [
            (old, new, what)
            for old, (new, what) in self._NER_TYPE_SPLIT.items()
            if old in enabled and new not in enabled
        ]
        if not moved:
            return
        self._ner_split_warned = True
        for old, new, what in moved:
            logger.warning(
                "%s no longer covers %s — those spans are reported as %s now, which "
                "is not enabled. Add it with add_entity(Entity.%s, action, "
                "layers=['ner']) to keep covering them, or ignore this if you only "
                "wanted %s itself.",
                old,
                what,
                new,
                new,
                old,
            )

    def _rebuild(self) -> None:
        """Rebuild detectors and engine when configuration changes."""
        self._maybe_warn_unsalted()
        self._detectors: list[BaseDetector] = []
        self._llm_detector: LLMDetector | None = None
        entity_cfg = self._config.get("entities", {})
        # Everything below that leaves a layer covering less than was configured
        # is recorded here as well as logged; the engine puts it on every result.
        build_warnings: list[str] = []

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
                    phone_regions=self._config.get("phone_regions") or None,
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
                        if resolved != model:
                            build_warnings.append(_ner_fallback_warning(model, resolved))
                        if resolved in loaded:  # avoid duplicate detectors
                            continue
                        self._detectors.append(NERDetector(enabled_ner, resolved))
                        loaded.add(resolved)
                    except Exception as exc:
                        logger.warning(
                            "SpaCy NER model %r could not be loaded, skipping it. Error: %s",
                            model,
                            exc,
                        )
                        build_warnings.append(
                            f"NERDetector did not run: SpaCy model {model!r} could not be "
                            f"loaded ({type(exc).__name__}: {exc})."
                        )
                if models and not loaded:
                    build_warnings.append(
                        "The NER layer has no model loaded and detects nothing, so "
                        f"{', '.join(sorted(enabled_ner))} will only be found if another "
                        "layer covers them."
                    )

        # LLM detector (optional). Kept on a dedicated attribute too, so the
        # semantic is_sensitive() check can reuse its backend directly.
        llm_cfg = self._config.get("llm_detector", {})
        if llm_cfg.get("enabled", False):
            self._llm_detector = self._build_llm_detector(llm_cfg)
            self._detectors.append(self._llm_detector)

        self._engine = DetectionEngine(self._config, self._detectors, build_warnings=build_warnings)
        if self._config.get("strict") and self._engine.build_warnings:
            raise DegradedScanError(list(self._engine.build_warnings))

    def _build_llm_detector(self, llm_cfg: dict[str, Any]) -> LLMDetector:
        """Build the LLM detector according to configuration."""
        from wardcat.detectors.llm_detector import LLMDetector
        from wardcat.llm.backends.registry import create_backend

        # Build one of the built-in backends selected by llm_cfg["backend"].
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
        from wardcat.llm.circuit import CircuitBreaker

        breaker = CircuitBreaker(
            llm_cfg.get("circuit_failures", 3), llm_cfg.get("circuit_cooldown", 30.0)
        )
        return LLMDetector(
            backend=backend,
            enabled_entities=enabled,
            timeout=timeout,
            cache_ttl=cache_ttl,
            breaker=breaker,
        )

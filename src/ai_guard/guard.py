from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_guard.config.loader import load_config
from ai_guard.core.engine import DetectionEngine
from ai_guard.core.models import ScanResult, warn_unknown_entity
from ai_guard.detectors.base import BaseDetector
from ai_guard.detectors.regex_detector import RegexDetector

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
            model, model, model,
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
        model, fallback, installed, model,
    )
    return fallback


# Central table mapping each entity to its detector
_REGEX_ENTITIES = {
    "CREDIT_CARD", "EMAIL", "PHONE", "IBAN", "IP_ADDRESS", "IPv6",
    "TC_ID", "ADDRESS", "POSTAL_CODE",
    "UUID", "SSN", "MAC_ADDRESS", "JWT", "NIN", "CUSTOM_SECRET",
    "UK_POSTAL_CODE", "US_ZIP_CODE", "EU_NATIONAL_ID",
    "PASSPORT", "CODICE_FISCALE", "DATE_OF_BIRTH",
}
_NER_ENTITIES   = {"PERSON", "ORG", "ADDRESS"}


class LLMGuard:
    """
    The main interface exposed to users.

    Programmatic API (method chaining)::

        guard = (
            LLMGuard(salt=os.environ["LLMGUARD_SALT"])
            .configure_entity("EMAIL",       enabled=True,  action="hash")
            .configure_entity("CREDIT_CARD", enabled=True,  action="hash")
            .configure_entity("ORG",         enabled=False)
        )
        result = guard.scan(text)

    Declarative API (YAML)::

        guard = LLMGuard(config_path="config/my_policy.yaml")
        result = guard.scan(text)

    Environment variables (override YAML and constructor arguments)::

        LLMGUARD_SALT          — hash salt value
        LLMGUARD_LLM_URL       — Ollama/OpenAI-compat service URL
        LLMGUARD_LLM_MODEL     — LLM model name
        LLMGUARD_LLM_API_KEY   — API key (OpenAI-compat)
        LLMGUARD_LLM_TIMEOUT   — LLM timeout (seconds, default 60)
        LLMGUARD_SPACY_MODEL   — SpaCy model name

    LLM detector (Ollama)::

        guard = LLMGuard(
            use_llm=True,
            llm_model="llama3.1:8b",
            llm_base_url="http://localhost:11434",
        )
        result = guard.scan(text)
    """

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        salt: str = "",
        use_ner: bool = True,
        spacy_model: str = "en_core_web_sm",
        use_llm: bool = False,
        llm_backend: str = "ollama",               # "ollama" | "openai_compatible" | "transformers"
        llm_model: str = "llama3.2",
        llm_base_url: str = "http://localhost:11434",
        llm_api_key: str = "",
        llm_timeout: int = 60,
        auto_pull: bool = False,          # Ollama: automatically download if model is missing
        llm_device_map: str = "auto",     # Transformers: GPU distribution
        llm_load_in_8bit: bool = False,   # Transformers: 8-bit quantization
        llm_load_in_4bit: bool = False,   # Transformers: 4-bit quantization
    ) -> None:
        self._config = load_config(config_path)

        # Constructor arguments override YAML
        # (environment variables were already applied inside load_config)
        if salt:
            self._config["salt"] = salt
        if not use_ner:
            self._config["use_ner"] = False
        if spacy_model != "en_core_web_sm":
            self._config["spacy_model"] = spacy_model

        # LLM detector overrides
        llm_cfg = self._config.setdefault("llm_detector", {})
        if use_llm:
            llm_cfg["enabled"] = True
        if llm_backend != "ollama":
            llm_cfg["backend"] = llm_backend
        if llm_model != "llama3.2":
            llm_cfg["model"] = llm_model
        if llm_base_url != "http://localhost:11434":
            llm_cfg["base_url"] = llm_base_url
        if llm_api_key:
            llm_cfg["api_key"] = llm_api_key
        if llm_timeout != 60:
            llm_cfg["timeout"] = llm_timeout
        if auto_pull:
            llm_cfg["auto_pull"] = True
        if llm_device_map != "auto":
            llm_cfg["device_map"] = llm_device_map
        if llm_load_in_8bit:
            llm_cfg["load_in_8bit"] = True
        if llm_load_in_4bit:
            llm_cfg["load_in_4bit"] = True

        # Salt warning: hash action is configured but salt is empty
        effective_salt = self._config.get("salt", "")
        if not effective_salt:
            entity_cfg = self._config.get("entities", {})
            has_hash = any(
                cfg.get("action") == "hash"
                for cfg in entity_cfg.values()
                if isinstance(cfg, dict)
            )
            if has_hash:
                logger.warning(
                    "Hash salt is empty — identical PII values will always produce the same hash. "
                    "Set the LLMGUARD_SALT environment variable in production."
                )

        self._rebuild()

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan(self, text: str) -> ScanResult:
        """Scan text and return a ScanResult."""
        return self._engine.scan(text)

    async def scan_async(self, text: str) -> ScanResult:
        """Async wrapper for :meth:`scan` — runs in a thread pool executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.scan, text)

    def scan_batch(
        self, texts: List[str], *, max_workers: Optional[int] = None
    ) -> List[ScanResult]:
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

        results: List[ScanResult | None] = [None] * len(texts)

        def _scan_one(idx: int, text: str) -> tuple[int, ScanResult]:
            try:
                return idx, self._engine.scan(text)
            except Exception as exc:
                logger.error(
                    "scan_batch item %d failed (%s: %s), returning original text.",
                    idx, type(exc).__name__, exc,
                )
                return idx, ScanResult(
                    original_text=text,
                    sanitized_text=text,
                    violations=[],
                    scan_error=f"{type(exc).__name__}: {exc}",
                )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_scan_one, i, text): i
                for i, text in enumerate(texts)
            }
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results  # type: ignore[return-value]

    async def scan_batch_async(
        self, texts: List[str], *, max_workers: Optional[int] = None
    ) -> List[ScanResult]:
        """Async wrapper for :meth:`scan_batch` — runs in a thread pool executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.scan_batch(texts, max_workers=max_workers)
        )

    # ------------------------------------------------------------------
    # Programmatic API
    # ------------------------------------------------------------------

    def configure_entity(
        self,
        entity_type: str,
        enabled: bool = True,
        action: str = "warn",
    ) -> "LLMGuard":
        """
        Configure a single entity type. Supports method chaining.

        :param entity_type: E.g. "EMAIL", "PERSON", "CREDIT_CARD"
        :param enabled:     Include this entity in the scan engine
        :param action:      "warn" or "hash"
        """
        warn_unknown_entity(entity_type)
        if action not in ("warn", "hash"):
            raise ValueError(
                f"Invalid action {action!r}. Valid values: 'warn', 'hash'"
            )
        self._config.setdefault("entities", {})[entity_type] = {
            "enabled": enabled,
            "action": action,
        }
        self._rebuild()
        return self

    def set_salt(self, salt: str) -> "LLMGuard":
        """Update the hash salt."""
        self._config["salt"] = salt
        self._rebuild()
        return self

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        """Rebuild detectors and engine when configuration changes."""
        self._detectors: List[BaseDetector] = []
        entity_cfg = self._config.get("entities", {})

        # Regex detector
        enabled_regex = {
            e for e in _REGEX_ENTITIES
            if entity_cfg.get(e, {}).get("enabled", True)
        }
        if enabled_regex:
            self._detectors.append(RegexDetector(enabled_regex))

        # SpaCy NER detector (optional)
        if self._config.get("use_ner", True):
            enabled_ner = {
                e for e in _NER_ENTITIES
                if entity_cfg.get(e, {}).get("enabled", True)
            }
            if enabled_ner:
                try:
                    from ai_guard.detectors.ner_detector import NERDetector
                    model = self._config.get("spacy_model", "en_core_web_sm")
                    model = _resolve_spacy_model(model)
                    self._detectors.append(NERDetector(enabled_ner, model))
                except Exception as exc:
                    logger.warning(
                        "SpaCy NER could not be loaded, using regex only. Error: %s", exc
                    )

        # LLM detector (optional)
        llm_cfg = self._config.get("llm_detector", {})
        if llm_cfg.get("enabled", False):
            self._detectors.append(self._build_llm_detector(llm_cfg))

        self._engine = DetectionEngine(self._config, self._detectors)

    def _build_llm_detector(self, llm_cfg: Dict[str, Any]) -> BaseDetector:
        """Build the LLM detector according to configuration."""
        from ai_guard.detectors.llm_detector import LLMDetector

        backend_name = llm_cfg.get("backend", "ollama")
        model        = llm_cfg.get("model",    "llama3.2")
        base_url     = llm_cfg.get("base_url", "http://localhost:11434")
        api_key      = llm_cfg.get("api_key",  "")
        timeout      = llm_cfg.get("timeout",  60)

        if backend_name == "ollama":
            from ai_guard.llm.backends.ollama import OllamaBackend
            from ai_guard.llm.model_manager import ModelManager
            ollama_backend = OllamaBackend(base_url=base_url, model=model)
            if llm_cfg.get("auto_pull", False):
                mgr = ModelManager(ollama_backend)
                mgr.ensure_available(model, verbose=True)
            backend = ollama_backend
        elif backend_name == "openai_compatible":
            from ai_guard.llm.backends.openai_compat import OpenAICompatBackend
            backend = OpenAICompatBackend(base_url=base_url, model=model, api_key=api_key)
        elif backend_name == "transformers":
            from ai_guard.llm.backends.transformers_backend import TransformersBackend
            backend = TransformersBackend(
                model=model,
                device_map=llm_cfg.get("device_map", "auto"),
                load_in_8bit=llm_cfg.get("load_in_8bit", False),
                load_in_4bit=llm_cfg.get("load_in_4bit", False),
            )
        else:
            raise ValueError(
                f"Unknown LLM backend: {backend_name!r}. "
                "Valid values: 'ollama', 'openai_compatible', 'transformers'"
            )

        entity_cfg = llm_cfg.get("entities", {})
        enabled = {
            e for e, cfg in entity_cfg.items()
            if cfg.get("enabled", True)
        }

        # Add LLM entity actions to global engine config (without overriding)
        for entity, cfg in entity_cfg.items():
            self._config["entities"].setdefault(entity, cfg)

        return LLMDetector(backend=backend, enabled_entities=enabled, timeout=timeout)

from __future__ import annotations

import concurrent.futures
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from wardcat.core.actions import registered_actions
from wardcat.exceptions import ConfigError
from wardcat.llm.backends.registry import registered_backends

logger = logging.getLogger(__name__)

# Library-internal default configuration.
# Values from a YAML file (if provided) override these (deep-merge).
# The library does not read environment variables.
DEFAULT_CONFIG: dict[str, Any] = {
    "salt": "",
    # NER is off by default and ships no default model — enable it explicitly via
    # Wardcat(language=...) / Wardcat(spacy_model=...) or a YAML use_ner + spacy_model.
    "use_ner": False,
    "scan_batch_workers": 4,  # thread pool size for scan_batch()
    "max_text_bytes": 500_000,  # maximum input size in bytes
    "custom_patterns": {},  # user-defined regex patterns
    "allowlist": [],  # exact values to never flag (e.g. ["no-reply@company.com"])
    "denylist": [],  # always-flag entries: [{value, entity_type}]
    # Propagation: redact every occurrence of a value once any layer detects it
    # (fills in repeats a model-based layer reports only once). Off by default.
    "propagate_matches": False,
    "propagate_min_length": 3,  # skip values shorter than this to avoid over-redaction
    # ── LLM detector configuration ────────────────────────────────────────
    "llm_detector": {
        "enabled": False,
        "backend": "ollama",  # "ollama" | "openai_compatible"
        "model": "llama3.2",
        "base_url": "http://localhost:11434",  # Ollama default; add /v1 for openai_compat
        "api_key": "",  # for openai_compat; empty for most on-prem
        "timeout": 60,
        "cache_ttl": 0,  # LLM response cache TTL in seconds; 0 = disabled
        "adjudicate": False,  # ensemble mode: LLM verifies regex/NER candidates
        "entities": {  # which types to query the LLM for
            "CREDIT_CARD": {"enabled": True, "action": "hash"},
            "EMAIL": {"enabled": True, "action": "warn"},
            "PHONE": {"enabled": True, "action": "warn"},
            "PERSON": {"enabled": True, "action": "hash"},
            "ORG": {"enabled": False, "action": "warn"},
            "ADDRESS": {"enabled": True, "action": "warn"},
            "IBAN": {"enabled": True, "action": "hash"},
            "TC_ID": {"enabled": True, "action": "hash"},
            "IP_ADDRESS": {"enabled": True, "action": "warn"},
            "IPv6": {"enabled": True, "action": "warn"},
            "UUID": {"enabled": True, "action": "warn"},
            "SSN": {"enabled": True, "action": "hash"},
            "MAC_ADDRESS": {"enabled": True, "action": "warn"},
            "JWT": {"enabled": True, "action": "hash"},
            "POSTAL_CODE": {"enabled": True, "action": "warn"},
            "NIN": {"enabled": True, "action": "hash"},
            "CUSTOM_SECRET": {"enabled": True, "action": "hash"},
            "PASSPORT": {"enabled": True, "action": "hash"},
            "EU_NATIONAL_ID": {"enabled": True, "action": "hash"},
            "UK_POSTAL_CODE": {"enabled": True, "action": "warn"},
            "US_ZIP_CODE": {"enabled": True, "action": "warn"},
            "CODICE_FISCALE": {"enabled": True, "action": "hash"},
            "VAT_NUMBER": {"enabled": True, "action": "warn"},
            # GDPR Art.9 special-category data — LLM-only, off by default
            "SPECIAL_CATEGORY": {"enabled": False, "action": "redact"},
        },
    },
    # ── GLiNER detector configuration ─────────────────────────────────────
    # Zero-shot transformer NER (optional; needs the ``gliner`` extra). Off by
    # default; enable with Wardcat().with_gliner(). Like NER, the entity types
    # it scans for are opt-in via add_entity(...).
    "gliner_detector": {
        "enabled": False,
        "model": "fastino/gliner2-privacy-filter-PII-multi",
        "threshold": 0.5,  # drop spans below this confidence
        "quantize": False,  # load a quantized model (less memory, slightly lower quality)
        "chunk_size": 1500,  # split longer input into windows (GLiNER truncates long text)
    },
    # NER/regex entities are opt-in: nothing is enabled by default.
    # Add what you need with Wardcat().add_entity(...) / add_entities(...),
    # or enable everything with add_entity(Entity.ALL, action=...).
    "entities": {},
}

# Valid backends come from the live registry (built-in + any registered by the user).
# Valid top-level config keys — used to catch typos in YAML files
_KNOWN_CONFIG_KEYS = frozenset(
    {
        "salt",
        "spacy_model",
        "spacy_models",
        "spacy_auto_download",
        "use_ner",
        "scan_batch_workers",
        "max_text_bytes",
        "custom_patterns",
        "llm_detector",
        "gliner_detector",
        "entities",
        "allowlist",
        "denylist",
        "propagate_matches",
        "propagate_min_length",
    }
)


def _check_redos(pattern: re.Pattern, timeout: float = 0.5) -> bool:
    """Return True if the pattern appears safe, False if it times out (potential ReDoS).

    Tests the compiled regex against a known pathological input (repeated 'a's followed
    by a non-matching character) to detect catastrophic backtracking at config load time.
    This prevents user-supplied custom patterns from locking up the server at runtime.

    Note: uses a thread with timeout; the thread may continue running briefly after
    the timeout, but the main thread is not blocked beyond the timeout window.
    """
    test_input = "a" * 50 + "b"
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(pattern.search, test_input)
        try:
            future.result(timeout=timeout)
            return True
        except concurrent.futures.TimeoutError:
            logger.warning("Pattern %r timed out during ReDoS check — rejecting.", pattern.pattern)
            return False


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Load configuration from defaults and an optional YAML file.

    Priority order (highest to lowest):
    1. YAML file (if path is provided)
    2. DEFAULT_CONFIG

    The library does **not** read environment variables — pass configuration
    explicitly via :class:`~wardcat.Wardcat` constructor arguments or a YAML
    file. (The ``wardcat`` CLI, being an application, does read ``WARDCAT_*``
    env vars as defaults.)

    If ``"default"`` is passed as ``path``, the bundled
    ``wardcat/config/default.yaml`` file is used.

    The result is validated; raises :class:`~wardcat.exceptions.ConfigError`
    (a subclass of ``ValueError``) if any value is invalid.
    """
    config = _deep_copy(DEFAULT_CONFIG)
    if path is not None:
        if str(path) == "default":
            # Load the default template bundled with the package
            from importlib.resources import files

            yaml_text = files("wardcat.config").joinpath("default.yaml").read_text(encoding="utf-8")
            user_config: dict[str, Any] = yaml.safe_load(yaml_text) or {}
        else:
            file_path = Path(path).resolve()
            if not file_path.exists():
                raise FileNotFoundError(f"Config file not found: {file_path}")
            with file_path.open("r", encoding="utf-8") as fh:
                user_config = yaml.safe_load(fh) or {}
        config = _deep_merge(config, user_config)

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """
    Validate the configuration. Raises
    :class:`~wardcat.exceptions.ConfigError` (a ``ValueError`` subclass) if any
    value is invalid.

    Each section is validated by a dedicated helper so this stays a thin
    coordinator (and each rule is independently readable/testable).
    """
    _warn_unknown_keys(config)
    _validate_entity_map(config.get("entities", {}), "entity")
    _validate_custom_patterns(config.get("custom_patterns", {}))
    _validate_allowlist(config.get("allowlist", []))
    _validate_denylist(config.get("denylist", []))
    _validate_llm_detector(config.get("llm_detector", {}))
    _validate_gliner_detector(config.get("gliner_detector", {}))


def _warn_unknown_keys(config: dict[str, Any]) -> None:
    unknown_keys = set(config.keys()) - _KNOWN_CONFIG_KEYS
    if unknown_keys:
        logger.warning(
            "Unknown configuration key(s): %s — check for typos. Valid top-level keys: %s",
            sorted(unknown_keys),
            sorted(_KNOWN_CONFIG_KEYS),
        )


def _validate_action(action: Any, where: str) -> None:
    if action not in registered_actions():
        raise ConfigError(
            f"Invalid action '{action}' ({where}). Valid values: {sorted(registered_actions())}"
        )


def _validate_entity_map(entities: dict[str, Any], label: str) -> None:
    if not isinstance(entities, dict):
        raise ConfigError(f"'{label} map' must be a dict, got {type(entities).__name__}.")
    for entity_name, entity_cfg in entities.items():
        if not isinstance(entity_cfg, dict):
            raise ConfigError(
                f"Invalid {label} configuration '{entity_name}': expected dict, "
                f"got {type(entity_cfg).__name__}."
            )
        _validate_action(entity_cfg.get("action", "warn"), f"{label}: {entity_name}")


def _validate_custom_patterns(custom_patterns: dict[str, Any]) -> None:
    for pattern_name, pattern_cfg in custom_patterns.items():
        if not isinstance(pattern_cfg, dict):
            raise ConfigError(
                f"Invalid custom_patterns entry '{pattern_name}': expected dict, "
                f"got {type(pattern_cfg).__name__}."
            )
        if "pattern" not in pattern_cfg:
            raise ConfigError(f"Custom pattern '{pattern_name}' is missing required 'pattern' key.")
        if not isinstance(pattern_cfg["pattern"], str):
            raise ConfigError(f"Custom pattern '{pattern_name}'.pattern must be a string.")
        _validate_action(pattern_cfg.get("action", "warn"), f"custom pattern: {pattern_name}")
        try:
            compiled = re.compile(pattern_cfg["pattern"])
        except re.error as exc:
            raise ConfigError(f"Custom pattern '{pattern_name}' has invalid regex: {exc}") from exc
        if not _check_redos(compiled):
            raise ConfigError(
                f"Custom pattern '{pattern_name}' may cause catastrophic backtracking "
                "(ReDoS). Simplify the pattern or remove nested quantifiers."
            )


def _validate_allowlist(allowlist: Any) -> None:
    if not isinstance(allowlist, list):
        raise ConfigError(f"'allowlist' must be a list of strings, got {type(allowlist).__name__}.")
    for item in allowlist:
        if not isinstance(item, str):
            raise ConfigError(
                f"Each 'allowlist' entry must be a string, got {type(item).__name__}: {item!r}"
            )


def _validate_denylist(denylist: Any) -> None:
    if not isinstance(denylist, list):
        raise ConfigError(f"'denylist' must be a list of dicts, got {type(denylist).__name__}.")
    for entry in denylist:
        if not isinstance(entry, dict):
            raise ConfigError(
                f"Each 'denylist' entry must be a dict with 'value' or 'pattern', "
                f"got {type(entry).__name__}: {entry!r}"
            )
        has_value = "value" in entry
        has_pattern = "pattern" in entry
        if not has_value and not has_pattern:
            raise ConfigError(
                f"Denylist entry must have either a 'value' or a 'pattern' key: {entry!r}"
            )
        if has_value and not isinstance(entry["value"], str):
            raise ConfigError(
                f"Denylist entry 'value' must be a string, got {type(entry['value']).__name__}"
            )
        if has_pattern:
            if not isinstance(entry["pattern"], str):
                raise ConfigError(
                    f"Denylist entry 'pattern' must be a string, "
                    f"got {type(entry['pattern']).__name__}"
                )
            try:
                re.compile(entry["pattern"])
            except re.error as exc:
                raise ConfigError(
                    f"Denylist entry 'pattern' {entry['pattern']!r} is not valid regex: {exc}"
                ) from exc


def _validate_llm_detector(llm_cfg: dict[str, Any]) -> None:
    backend = llm_cfg.get("backend", "ollama")
    valid_backends = registered_backends()
    if backend not in valid_backends:
        raise ConfigError(
            f"Invalid LLM backend '{backend}'. Registered backends: {sorted(valid_backends)}"
        )

    timeout = llm_cfg.get("timeout", 60)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ConfigError(f"Invalid llm_detector.timeout: {timeout!r} (must be a positive number)")

    _validate_entity_map(llm_cfg.get("entities", {}), "llm_detector.entities")


def _validate_gliner_detector(gliner_cfg: dict[str, Any]) -> None:
    if not isinstance(gliner_cfg, dict):
        raise ConfigError(f"'gliner_detector' must be a dict, got {type(gliner_cfg).__name__}.")
    # Absent section is valid (the detector is off and falls back to defaults);
    # only an explicitly bad value is rejected.
    model = gliner_cfg.get("model", "fastino/gliner2-privacy-filter-PII-multi")
    if not isinstance(model, str) or not model:
        raise ConfigError("gliner_detector.model must be a non-empty string.")
    threshold = gliner_cfg.get("threshold", 0.5)
    if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
        raise ConfigError(
            f"Invalid gliner_detector.threshold: {threshold!r} (must be a number in [0, 1])."
        )
    chunk_size = gliner_cfg.get("chunk_size", 1500)
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ConfigError(
            f"Invalid gliner_detector.chunk_size: {chunk_size!r} (must be a positive integer)."
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively applies override on top of base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _deep_copy(obj: Any) -> Any:
    """A simple recursive copy instead of stdlib copy.deepcopy."""
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_copy(i) for i in obj]
    return obj

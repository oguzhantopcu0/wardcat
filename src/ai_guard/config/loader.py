from __future__ import annotations

import concurrent.futures
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from ai_guard.exceptions import ConfigError
from ai_guard.llm.backends.base import Backend

logger = logging.getLogger(__name__)

# Library-internal default configuration.
# Values from a YAML file (if provided) override these (deep-merge).
# The library does not read environment variables.
DEFAULT_CONFIG: dict[str, Any] = {
    "salt": "",
    # NER is off by default and ships no default model — enable it explicitly via
    # AIGuard(language=...) / AIGuard(spacy_model=...) or a YAML use_ner + spacy_model.
    "use_ner": False,
    "scan_batch_workers": 4,  # thread pool size for scan_batch()
    "max_text_bytes": 500_000,  # maximum input size in bytes
    "custom_patterns": {},  # user-defined regex patterns
    "allowlist": [],  # exact values to never flag (e.g. ["no-reply@company.com"])
    "denylist": [],  # always-flag entries: [{value, entity_type}]
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
    # NER/regex entities are opt-in: nothing is enabled by default.
    # Add what you need with AIGuard().add_entity(...) / add_entities(...),
    # or enable everything with add_entity(Entity.ALL, action=...).
    "entities": {},
}

# Valid action values
_VALID_ACTIONS = {"warn", "hash", "mask", "redact"}
# Valid backend values
_VALID_BACKENDS = {b.value for b in Backend}
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
        "entities",
        "allowlist",
        "denylist",
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
    explicitly via :class:`~ai_guard.AIGuard` constructor arguments or a YAML
    file. (The ``ai-guard`` CLI, being an application, does read ``AIGUARD_*``
    env vars as defaults.)

    If ``"default"`` is passed as ``path``, the bundled
    ``ai_guard/config/default.yaml`` file is used.

    The result is validated; raises :class:`~ai_guard.exceptions.ConfigError`
    (a subclass of ``ValueError``) if any value is invalid.
    """
    config = _deep_copy(DEFAULT_CONFIG)
    if path is not None:
        if str(path) == "default":
            # Load the default template bundled with the package
            from importlib.resources import files

            yaml_text = (
                files("ai_guard.config").joinpath("default.yaml").read_text(encoding="utf-8")
            )
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
    :class:`~ai_guard.exceptions.ConfigError` (a ``ValueError`` subclass) if any
    value is invalid.
    """
    unknown_keys = set(config.keys()) - _KNOWN_CONFIG_KEYS
    if unknown_keys:
        logger.warning(
            "Unknown configuration key(s): %s — check for typos. Valid top-level keys: %s",
            sorted(unknown_keys),
            sorted(_KNOWN_CONFIG_KEYS),
        )

    entities = config.get("entities", {})
    for entity_name, entity_cfg in entities.items():
        if not isinstance(entity_cfg, dict):
            raise ConfigError(
                f"Invalid entity configuration '{entity_name}': expected dict, "
                f"got {type(entity_cfg).__name__}."
            )
        action = entity_cfg.get("action", "warn")
        if action not in _VALID_ACTIONS:
            raise ConfigError(
                f"Invalid action '{action}' (entity: {entity_name}). "
                f"Valid values: {sorted(_VALID_ACTIONS)}"
            )

    custom_patterns = config.get("custom_patterns", {})
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
        action = pattern_cfg.get("action", "warn")
        if action not in _VALID_ACTIONS:
            raise ConfigError(
                f"Invalid action '{action}' for custom pattern '{pattern_name}'. "
                f"Valid values: {sorted(_VALID_ACTIONS)}"
            )
        try:
            compiled = re.compile(pattern_cfg["pattern"])
        except re.error as exc:
            raise ConfigError(f"Custom pattern '{pattern_name}' has invalid regex: {exc}") from exc
        if not _check_redos(compiled):
            raise ConfigError(
                f"Custom pattern '{pattern_name}' may cause catastrophic backtracking "
                "(ReDoS). Simplify the pattern or remove nested quantifiers."
            )

    allowlist = config.get("allowlist", [])
    if not isinstance(allowlist, list):
        raise ConfigError(f"'allowlist' must be a list of strings, got {type(allowlist).__name__}.")
    for item in allowlist:
        if not isinstance(item, str):
            raise ConfigError(
                f"Each 'allowlist' entry must be a string, got {type(item).__name__}: {item!r}"
            )

    denylist = config.get("denylist", [])
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

    llm_cfg = config.get("llm_detector", {})
    backend = llm_cfg.get("backend", "ollama")
    if backend not in _VALID_BACKENDS:
        raise ConfigError(
            f"Invalid LLM backend '{backend}'. Valid values: {sorted(_VALID_BACKENDS)}"
        )

    timeout = llm_cfg.get("timeout", 60)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ConfigError(f"Invalid llm_detector.timeout: {timeout!r} (must be a positive number)")

    llm_entities = llm_cfg.get("entities", {})
    if not isinstance(llm_entities, dict):
        raise ConfigError(
            f"'llm_detector.entities' must be a dict, got {type(llm_entities).__name__}."
        )
    for entity_name, entity_cfg in llm_entities.items():
        if not isinstance(entity_cfg, dict):
            raise ConfigError(
                f"Invalid llm_detector.entities['{entity_name}']: expected dict, "
                f"got {type(entity_cfg).__name__}."
            )
        action = entity_cfg.get("action", "warn")
        if action not in _VALID_ACTIONS:
            raise ConfigError(
                f"Invalid action '{action}' (llm_detector.entities['{entity_name}']). "
                f"Valid values: {sorted(_VALID_ACTIONS)}"
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

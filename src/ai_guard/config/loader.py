from __future__ import annotations

import concurrent.futures
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# Library-internal default configuration.
# Values from YAML override these (deep-merge).
# Environment variables (LLMGUARD_*) override everything.
DEFAULT_CONFIG: Dict[str, Any] = {
    "salt": "",
    "spacy_model": "en_core_web_sm",
    "use_ner": True,
    "scan_batch_workers": 4,   # thread pool size for scan_batch()
    "max_text_bytes": 500_000,  # maximum input size in bytes
    "custom_patterns": {},      # user-defined regex patterns
    "allowlist": [],            # exact values to never flag (e.g. ["no-reply@company.com"])
    "denylist":  [],            # always-flag entries: [{value, entity_type}]
    # ── LLM detector configuration ────────────────────────────────────────
    "llm_detector": {
        "enabled":  False,
        "backend":  "ollama",                   # "ollama" | "openai_compatible"
        "model":    "llama3.2",
        "base_url": "http://localhost:11434",   # Ollama default; add /v1 for openai_compat
        "api_key":  "",                         # for openai_compat; empty for most on-prem
        "timeout":   60,
        "cache_ttl": 0,                          # LLM response cache TTL in seconds; 0 = disabled
        "adjudicate": False,                      # ensemble mode: LLM verifies regex/NER candidates
        "entities": {                           # which types to query the LLM for
            "CREDIT_CARD":  {"enabled": True,  "action": "hash"},
            "EMAIL":        {"enabled": True,  "action": "warn"},
            "PHONE":        {"enabled": True,  "action": "warn"},
            "PERSON":       {"enabled": True,  "action": "hash"},
            "ORG":          {"enabled": False, "action": "warn"},
            "ADDRESS":      {"enabled": True,  "action": "warn"},
            "IBAN":         {"enabled": True,  "action": "hash"},
            "TC_ID":        {"enabled": True,  "action": "hash"},
            "IP_ADDRESS":   {"enabled": True,  "action": "warn"},
            "IPv6":         {"enabled": True,  "action": "warn"},
            "UUID":         {"enabled": True,  "action": "warn"},
            "SSN":          {"enabled": True,  "action": "hash"},
            "MAC_ADDRESS":  {"enabled": True,  "action": "warn"},
            "JWT":          {"enabled": True,  "action": "hash"},
            "POSTAL_CODE":    {"enabled": True,  "action": "warn"},
            "NIN":            {"enabled": True,  "action": "hash"},
            "CUSTOM_SECRET":  {"enabled": True,  "action": "hash"},
            "PASSPORT":       {"enabled": True,  "action": "hash"},
            "EU_NATIONAL_ID": {"enabled": True,  "action": "hash"},
            "UK_POSTAL_CODE": {"enabled": True,  "action": "warn"},
            "US_ZIP_CODE":    {"enabled": True,  "action": "warn"},
            "CODICE_FISCALE": {"enabled": True,  "action": "hash"},
            "VAT_NUMBER":     {"enabled": True,  "action": "warn"},
        },
    },
    "entities": {
        # Regex-based
        "CREDIT_CARD": {"enabled": True,  "action": "hash"},
        "EMAIL":        {"enabled": True,  "action": "warn"},
        "PHONE":        {"enabled": True,  "action": "warn"},
        "IBAN":         {"enabled": True,  "action": "hash"},
        "IP_ADDRESS":   {"enabled": True,  "action": "warn"},
        "TC_ID":        {"enabled": True,  "action": "hash"},
        # Regex-based — address
        "ADDRESS":      {"enabled": True,  "action": "warn"},
        "POSTAL_CODE":  {"enabled": True,  "action": "warn"},
        # Regex-based — global identity/technical
        "UUID":         {"enabled": True,  "action": "warn"},
        "SSN":          {"enabled": True,  "action": "hash"},
        "MAC_ADDRESS":  {"enabled": True,  "action": "warn"},
        "JWT":          {"enabled": True,  "action": "hash"},
        "IPv6":         {"enabled": True,  "action": "warn"},
        "NIN":          {"enabled": True,  "action": "hash"},
        # SpaCy NER-based
        "PERSON":       {"enabled": True,  "action": "hash"},
        "ORG":          {"enabled": False, "action": "warn"},
        # Date detection
        "DATE_OF_BIRTH": {"enabled": True, "action": "hash"},
        # Turkish vehicle plate
        "VEHICLE_PLATE": {"enabled": True, "action": "warn"},
        # Financial amounts — disabled by default; enable for confidential document scanning
        "FINANCIAL_AMOUNT": {"enabled": False, "action": "redact"},
        # VAT / tax numbers (EU country-prefixed + Turkish Vergi No)
        "VAT_NUMBER": {"enabled": True, "action": "warn"},
    },
}

# Supported environment variables
_ENV_VARS = {
    "LLMGUARD_SALT":          ("salt",),
    "LLMGUARD_SPACY_MODEL":   ("spacy_model",),
    "LLMGUARD_LLM_URL":       ("llm_detector", "base_url"),
    "LLMGUARD_LLM_MODEL":     ("llm_detector", "model"),
    "LLMGUARD_LLM_API_KEY":   ("llm_detector", "api_key"),
    "LLMGUARD_LLM_TIMEOUT":   ("llm_detector", "timeout"),
}

# Valid action values
_VALID_ACTIONS = {"warn", "hash", "mask", "redact"}
# Valid backend values
_VALID_BACKENDS = {"ollama", "openai_compatible", "transformers"}
# Valid top-level config keys — used to catch typos in YAML files
_KNOWN_CONFIG_KEYS = frozenset({
    "salt", "spacy_model", "use_ner", "scan_batch_workers",
    "max_text_bytes", "custom_patterns",
    "llm_detector", "entities",
    "allowlist", "denylist",
})


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
            logger.warning(
                "Pattern %r timed out during ReDoS check — rejecting.", pattern.pattern
            )
            return False


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Load configuration and apply environment variables.

    Priority order (highest to lowest):
    1. Environment variables (LLMGUARD_*)
    2. YAML file (if path is provided)
    3. DEFAULT_CONFIG

    If ``"default"`` is passed as ``path``, the bundled
    ``ai_guard/config/default.yaml`` file is used.

    The result is validated; raises ValueError if there are errors.
    """
    config = _deep_copy(DEFAULT_CONFIG)
    if path is not None:
        if str(path) == "default":
            # Load the default template bundled with the package
            from importlib.resources import files
            yaml_text = files("ai_guard.config").joinpath("default.yaml").read_text(encoding="utf-8")
            user_config: Dict[str, Any] = yaml.safe_load(yaml_text) or {}
        else:
            file_path = Path(path).resolve()
            if not file_path.exists():
                raise FileNotFoundError(f"Config file not found: {file_path}")
            with file_path.open("r", encoding="utf-8") as fh:
                user_config = yaml.safe_load(fh) or {}
        config = _deep_merge(config, user_config)

    _apply_env_overrides(config)
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the configuration. Raises ValueError if any value is invalid.
    """
    unknown_keys = set(config.keys()) - _KNOWN_CONFIG_KEYS
    if unknown_keys:
        logger.warning(
            "Unknown configuration key(s): %s — check for typos. "
            "Valid top-level keys: %s",
            sorted(unknown_keys), sorted(_KNOWN_CONFIG_KEYS),
        )

    entities = config.get("entities", {})
    for entity_name, entity_cfg in entities.items():
        if not isinstance(entity_cfg, dict):
            raise ValueError(
                f"Invalid entity configuration '{entity_name}': expected dict, "
                f"got {type(entity_cfg).__name__}."
            )
        action = entity_cfg.get("action", "warn")
        if action not in _VALID_ACTIONS:
            raise ValueError(
                f"Invalid action '{action}' (entity: {entity_name}). "
                f"Valid values: {sorted(_VALID_ACTIONS)}"
            )

    custom_patterns = config.get("custom_patterns", {})
    for pattern_name, pattern_cfg in custom_patterns.items():
        if not isinstance(pattern_cfg, dict):
            raise ValueError(
                f"Invalid custom_patterns entry '{pattern_name}': expected dict, "
                f"got {type(pattern_cfg).__name__}."
            )
        if "pattern" not in pattern_cfg:
            raise ValueError(
                f"Custom pattern '{pattern_name}' is missing required 'pattern' key."
            )
        if not isinstance(pattern_cfg["pattern"], str):
            raise ValueError(
                f"Custom pattern '{pattern_name}'.pattern must be a string."
            )
        action = pattern_cfg.get("action", "warn")
        if action not in _VALID_ACTIONS:
            raise ValueError(
                f"Invalid action '{action}' for custom pattern '{pattern_name}'. "
                f"Valid values: {sorted(_VALID_ACTIONS)}"
            )
        try:
            compiled = re.compile(pattern_cfg["pattern"])
        except re.error as exc:
            raise ValueError(
                f"Custom pattern '{pattern_name}' has invalid regex: {exc}"
            )
        if not _check_redos(compiled):
            raise ValueError(
                f"Custom pattern '{pattern_name}' may cause catastrophic backtracking "
                "(ReDoS). Simplify the pattern or remove nested quantifiers."
            )

    allowlist = config.get("allowlist", [])
    if not isinstance(allowlist, list):
        raise ValueError(
            f"'allowlist' must be a list of strings, got {type(allowlist).__name__}."
        )
    for item in allowlist:
        if not isinstance(item, str):
            raise ValueError(
                f"Each 'allowlist' entry must be a string, got {type(item).__name__}: {item!r}"
            )

    denylist = config.get("denylist", [])
    if not isinstance(denylist, list):
        raise ValueError(
            f"'denylist' must be a list of dicts, got {type(denylist).__name__}."
        )
    for entry in denylist:
        if not isinstance(entry, dict):
            raise ValueError(
                f"Each 'denylist' entry must be a dict with 'value' or 'pattern', "
                f"got {type(entry).__name__}: {entry!r}"
            )
        has_value   = "value"   in entry
        has_pattern = "pattern" in entry
        if not has_value and not has_pattern:
            raise ValueError(
                f"Denylist entry must have either a 'value' or a 'pattern' key: {entry!r}"
            )
        if has_value and not isinstance(entry["value"], str):
            raise ValueError(
                f"Denylist entry 'value' must be a string, got {type(entry['value']).__name__}"
            )
        if has_pattern:
            if not isinstance(entry["pattern"], str):
                raise ValueError(
                    f"Denylist entry 'pattern' must be a string, "
                    f"got {type(entry['pattern']).__name__}"
                )
            try:
                re.compile(entry["pattern"])
            except re.error as exc:
                raise ValueError(
                    f"Denylist entry 'pattern' {entry['pattern']!r} is not valid regex: {exc}"
                )

    llm_cfg = config.get("llm_detector", {})
    backend = llm_cfg.get("backend", "ollama")
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Invalid LLM backend '{backend}'. "
            f"Valid values: {sorted(_VALID_BACKENDS)}"
        )

    timeout = llm_cfg.get("timeout", 60)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(f"Invalid llm_detector.timeout: {timeout!r} (must be a positive number)")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _apply_env_overrides(config: Dict[str, Any]) -> None:
    """
    Apply LLMGUARD_* environment variables to the config.
    When an environment variable is present, it always overrides the YAML/default value.
    """
    for env_var, key_path in _ENV_VARS.items():
        value = os.environ.get(env_var)
        if value is None:
            continue

        # Integer conversion for timeout
        if key_path == ("llm_detector", "timeout"):
            try:
                value = int(value)
            except ValueError:
                logger.warning(
                    "LLMGUARD_LLM_TIMEOUT has invalid value %r — ignored.", value
                )
                continue

        # Follow nested key path
        target = config
        for key in key_path[:-1]:
            target = target.setdefault(key, {})
        target[key_path[-1]] = value
        logger.debug("Read from environment variable: %s → %s", env_var, key_path)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
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

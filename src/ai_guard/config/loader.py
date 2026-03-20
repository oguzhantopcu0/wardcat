from __future__ import annotations

import logging
import os
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
    # ── LLM detector configuration ────────────────────────────────────────
    "llm_detector": {
        "enabled":  False,
        "backend":  "ollama",                   # "ollama" | "openai_compatible"
        "model":    "llama3.2",
        "base_url": "http://localhost:11434",   # Ollama default; add /v1 for openai_compat
        "api_key":  "",                         # for openai_compat; empty for most on-prem
        "timeout":  60,
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
            "POSTAL_CODE":  {"enabled": True,  "action": "warn"},
            "NIN":          {"enabled": True,  "action": "hash"},
            "CUSTOM_SECRET":{"enabled": False, "action": "hash"},
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
_VALID_ACTIONS = {"warn", "hash"}
# Valid backend values
_VALID_BACKENDS = {"ollama", "openai_compatible", "transformers"}


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
            file_path = Path(path)
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

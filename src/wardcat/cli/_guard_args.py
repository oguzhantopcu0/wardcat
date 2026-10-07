"""The options that build a guard, shared by ``scan``, ``check`` and later the hooks."""

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING

from wardcat.exceptions import ConfigError

if TYPE_CHECKING:  # pragma: no cover
    from wardcat import Wardcat

# Actions whose output depends on the salt. With no salt they still run, but a
# low-entropy value (a card, a TC number) hashed without one can be recovered by
# brute force — say so rather than let it pass unnoticed.
_SALTED_ACTIONS = frozenset({"hash", "tokenize", "surrogate"})


def add_guard_args(parser: argparse.ArgumentParser, *, default_action: str = "redact") -> None:
    scope = parser.add_argument_group("what to look for")
    scope.add_argument("--config", metavar="YAML", help="policy file (see the YAML reference)")
    scope.add_argument("--preset", metavar="NAME", help="a starting policy, e.g. kvkk")
    scope.add_argument(
        "--entity",
        action="append",
        default=[],
        metavar="TYPE[=ACTION][,...]",
        help="enable entities; repeatable, or comma-separated. Each item takes its own "
        "=ACTION; one given without uses --action.",
    )
    scope.add_argument(
        "--group",
        action="append",
        default=[],
        metavar="NAME",
        help="enable a whole entity group (core, financial, turkish, european, uk, us, "
        "network, identity, all); repeatable",
    )
    scope.add_argument(
        "--action",
        default=default_action,
        help=f"action for --entity and --group values given without one (default: {default_action})",
    )

    layers = parser.add_argument_group("layers")
    ner = layers.add_mutually_exclusive_group()
    ner.add_argument("--ner", metavar="MODEL", help="enable the NER layer with this SpaCy model")
    ner.add_argument(
        "--ner-language",
        metavar="LANG",
        help="enable the NER layer with the catalog model for a language (e.g. tr); "
        "the model must already be installed",
    )
    layers.add_argument("--llm", metavar="MODEL", help="enable the LLM layer with this model")
    add_llm_args(layers)
    layers.add_argument(
        "--adjudicate", action="store_true", help="let the LLM confirm the other layers' finds"
    )

    run = parser.add_argument_group("running")
    run.add_argument("--strict", action="store_true", help="refuse a degraded scan (exit 3)")
    run.add_argument(
        "--salt-env", metavar="VAR", help="environment variable holding the hashing salt"
    )


def add_llm_args(group: argparse._ArgumentGroup | argparse.ArgumentParser) -> None:
    group.add_argument(
        "--llm-backend",
        default="ollama",
        metavar="NAME",
        help="ollama (default), vllm, openai_compatible or transformers",
    )
    group.add_argument("--llm-base-url", metavar="URL", help="the backend's address")
    group.add_argument(
        "--llm-api-key-env", metavar="VAR", help="environment variable holding the backend's key"
    )
    group.add_argument(
        "--allow-http",
        action="store_true",
        help="allow plain HTTP to a remote backend (loopback never needs it)",
    )


def _env(var: str, what: str) -> str:
    value = os.environ.get(var, "")
    if not value:
        raise ConfigError(f"environment variable {var} is not set or empty ({what})")
    return value


def parse_entity_specs(specs: list[str], default_action: str) -> list[tuple[str, str]]:
    """``["EMAIL", "IBAN,CREDIT_CARD=mask"]`` → ``[("EMAIL", a), ("IBAN", a), ("CREDIT_CARD", "mask")]``.

    Each comma-separated item carries its own ``=ACTION``. An empty item is a
    usage error, not something to skip.
    """
    pairs: list[tuple[str, str]] = []
    for spec in specs:
        for item in spec.split(","):
            name, sep, action = item.strip().partition("=")
            name, action = name.strip(), action.strip()
            if not name or (sep and not action):
                raise ConfigError(f"--entity {spec!r} has an empty item")
            pairs.append((name, action if sep else default_action))
    return pairs


def llm_kwargs(args: argparse.Namespace) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "backend": args.llm_backend,
        "allow_http": args.allow_http,
    }
    if args.llm_base_url:
        kwargs["base_url"] = args.llm_base_url
    if args.llm_api_key_env:
        kwargs["api_key"] = _env(args.llm_api_key_env, "--llm-api-key-env")
    return kwargs


def build_guard(args: argparse.Namespace, *, scope_required: bool = True) -> Wardcat:
    """The guard the options describe.

    With *scope_required* (``scan``) a guard that would look for nothing is a
    usage error. Without it (``check``) the caller supplies a default scope.
    """
    from wardcat import Wardcat, entity_groups

    salt = _env(args.salt_env, "--salt-env") if args.salt_env else ""
    guard = Wardcat(config_path=args.config, salt=salt)
    if args.preset:
        guard.with_preset(args.preset)
    for group in args.group:
        try:
            names = getattr(entity_groups, f"{group}_entities")()
        except AttributeError:
            raise ConfigError(f"unknown --group {group!r}") from None
        guard.add_entities(sorted(names), action=args.action)
    for name, action in parse_entity_specs(args.entity, args.action):
        guard.add_entity(name, action)
    if scope_required and not (args.config or args.preset or args.entity or args.group):
        raise ConfigError("nothing to scan for: pass --preset, --entity, --group or --config")
    if args.ner:
        guard.with_ner(spacy_model=args.ner, auto_download=False)
    elif args.ner_language:
        guard.with_ner(language=args.ner_language, auto_download=False)
    if args.llm:
        guard.with_llm(model=args.llm, adjudicate=args.adjudicate, **llm_kwargs(args))  # type: ignore[arg-type]
    elif args.adjudicate:
        raise ConfigError("--adjudicate needs the LLM layer: pass --llm MODEL")
    if args.strict:
        guard.with_strict()
    _warn_on_unsalted(guard, salt)
    return guard


def _warn_on_unsalted(guard: Wardcat, salt: str) -> None:
    if salt:
        return
    salted = sorted(
        name for name, action in guard.entity_policy().items() if action in _SALTED_ACTIONS
    )
    if not salted:
        return
    hint = (
        " WARDCAT_SALT is set in the environment; pass --salt-env WARDCAT_SALT to use it."
        if os.environ.get("WARDCAT_SALT")
        else ""
    )
    print(
        f"warning: no salt, so {', '.join(salted)} are hashed without one and a "
        f"low-entropy value can be recovered by brute force; pass --salt-env VAR.{hint}",
        file=sys.stderr,
    )

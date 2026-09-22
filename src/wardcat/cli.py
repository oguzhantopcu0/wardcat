"""The ``wardcat`` command.

::

    wardcat scan [FILE|-] [--config policy.yaml] [--preset kvkk] [--entity EMAIL=redact ...]
                 [--action redact] [--ner MODEL] [--llm MODEL] [--strict] [--json]
                 [--salt-env WARDCAT_SALT]
    wardcat check-config policy.yaml
    wardcat entities [--layer regex|ner|llm]

Exit codes: ``0`` the text is clean, ``1`` it is not, ``2`` a configuration or
usage error, ``3`` the scan was degraded and ``--strict`` refused it.

The salt is never taken on the command line — a process list and a shell
history would keep it. ``--salt-env`` names the environment variable to read.
Output is the sanitized text, or with ``--json`` the ``redacted()`` dict; the
original text and the values found are never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

from wardcat.exceptions import ConfigError, DegradedScanError, WardcatError

if TYPE_CHECKING:  # pragma: no cover
    from wardcat import Wardcat

EXIT_CLEAN, EXIT_FOUND, EXIT_CONFIG, EXIT_DEGRADED = 0, 1, 2, 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wardcat", description="PII detection and anonymization.")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="scan a file or stdin and print the sanitized text")
    scan.add_argument("file", nargs="?", default="-", help="path to read, or - for stdin")
    scan.add_argument("--config", metavar="YAML", help="policy file (see the YAML reference)")
    scan.add_argument("--preset", metavar="NAME", help="a starting policy, e.g. kvkk")
    scan.add_argument(
        "--entity",
        action="append",
        default=[],
        metavar="TYPE[=ACTION]",
        help="enable one entity; repeatable. Without =ACTION, --action is used.",
    )
    scan.add_argument(
        "--action", default="redact", help="action for --entity values given without one"
    )
    scan.add_argument("--ner", metavar="MODEL", help="enable the NER layer with this SpaCy model")
    scan.add_argument("--llm", metavar="MODEL", help="enable the LLM layer with this Ollama model")
    scan.add_argument("--strict", action="store_true", help="refuse a degraded scan (exit 3)")
    scan.add_argument("--json", action="store_true", help="print the PII-free redacted() dict")
    scan.add_argument(
        "--salt-env", metavar="VAR", help="environment variable holding the hashing salt"
    )

    check = sub.add_parser("check-config", help="validate a policy file, including its patterns")
    check.add_argument("file", help="the YAML policy to check")

    entities = sub.add_parser("entities", help="list the entity types wardcat can detect")
    entities.add_argument("--layer", choices=["regex", "ner", "llm"])
    return parser


def _build_guard(args: argparse.Namespace) -> Wardcat:
    from wardcat import Wardcat

    salt = ""
    if args.salt_env:
        salt = os.environ.get(args.salt_env, "")
        if not salt:
            raise ConfigError(f"environment variable {args.salt_env} is not set or empty")
    guard = Wardcat(config_path=args.config, salt=salt)
    if args.preset:
        guard.with_preset(args.preset)
    for spec in args.entity:
        name, sep, action = spec.partition("=")
        guard.add_entity(name.strip(), action.strip() if sep else args.action)
    if not (args.config or args.preset or args.entity):
        raise ConfigError("nothing to scan for: pass --preset, --entity or --config")
    if args.ner:
        guard.with_ner(spacy_model=args.ner, auto_download=False)
    if args.llm:
        guard.with_llm(model=args.llm)
    if args.strict:
        guard.with_strict()
    return guard


def _read(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _scan(args: argparse.Namespace) -> int:
    guard = _build_guard(args)
    text = _read(args.file)
    result = guard.scan(text)
    if args.json:
        print(json.dumps(result.redacted(), ensure_ascii=False))
    else:
        sys.stdout.write(result.sanitized_text)
        if not result.sanitized_text.endswith("\n"):
            sys.stdout.write("\n")
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return EXIT_CLEAN if result.is_clean else EXIT_FOUND


def _check_config(args: argparse.Namespace) -> int:
    from wardcat import Wardcat

    # Loading runs every validation the library has: unknown keys, actions,
    # entity specs, and the ReDoS screen on custom and denylist patterns.
    guard = Wardcat(config_path=args.file)
    warnings = list(guard._engine.build_warnings)
    policy = guard.entity_policy()
    print(f"{args.file}: ok, {len(policy)} entity type(s) enabled")
    for warning in warnings:
        print(f"warning: {warning}")
    return EXIT_CLEAN


def _entities(args: argparse.Namespace) -> int:
    from wardcat import Wardcat

    for name in sorted(Wardcat.supported_entities(args.layer)):
        print(name)
    return EXIT_CLEAN


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "scan":
            return _scan(args)
        if args.command == "check-config":
            return _check_config(args)
        return _entities(args)
    except DegradedScanError as exc:
        for warning in exc.warnings:
            print(f"error: {warning}", file=sys.stderr)
        return EXIT_DEGRADED
    except (ConfigError, WardcatError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

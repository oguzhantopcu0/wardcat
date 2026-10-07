"""``wardcat check-config``: load a policy file through every validation."""

from __future__ import annotations

import argparse


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "check-config",
        help="validate a policy file, including its patterns",
        allow_abbrev=False,
    )
    p.add_argument("file", help="the YAML policy to check")
    p.set_defaults(run=run)


def run(args: argparse.Namespace) -> int:
    from wardcat import Wardcat
    from wardcat.cli import EXIT_CLEAN

    # Loading runs every validation the library has: unknown keys, actions,
    # entity specs, and the ReDoS screen on custom and denylist patterns.
    guard = Wardcat(config_path=args.file)
    warnings = list(guard._engine.build_warnings)
    policy = guard.entity_policy()
    print(f"{args.file}: ok, {len(policy)} entity type(s) enabled")
    for warning in warnings:
        print(f"warning: {warning}")
    return EXIT_CLEAN

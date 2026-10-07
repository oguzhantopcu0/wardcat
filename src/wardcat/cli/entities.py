"""``wardcat entities``: list what can be detected."""

from __future__ import annotations

import argparse


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "entities", help="list the entity types wardcat can detect", allow_abbrev=False
    )
    p.add_argument("--layer", choices=["regex", "ner", "llm"])
    p.set_defaults(run=run)


def run(args: argparse.Namespace) -> int:
    from wardcat import Wardcat
    from wardcat.cli import EXIT_CLEAN

    for name in sorted(Wardcat.supported_entities(args.layer)):
        print(name)
    return EXIT_CLEAN

"""``wardcat check-config``: load a policy file through every validation."""

from __future__ import annotations

import argparse
import logging
import sys


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "check-config",
        help="validate a policy file, including its patterns",
        allow_abbrev=False,
    )
    p.add_argument("file", help="the YAML policy to check")
    p.set_defaults(run=run)


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def run(args: argparse.Namespace) -> int:
    from wardcat import Wardcat
    from wardcat.cli import EXIT_CLEAN

    # Loading runs every validation the library has: unknown keys, actions,
    # entity specs, and the ReDoS screen on custom and denylist patterns. What
    # the loader only warns about (an unknown top-level key, a secret written
    # into the file) is collected and printed too.
    collector = _Collect()
    log = logging.getLogger("wardcat")
    log.addHandler(collector)
    try:
        guard = Wardcat(config_path=args.file)
    finally:
        log.removeHandler(collector)
    policy = guard.entity_policy()
    print(f"{args.file}: ok, {len(policy)} entity type(s) enabled")
    for warning in [*collector.messages, *guard._engine.build_warnings]:
        print(f"warning: {warning}", file=sys.stderr)
    return EXIT_CLEAN

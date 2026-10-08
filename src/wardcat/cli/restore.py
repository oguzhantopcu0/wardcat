"""``wardcat restore``: put the values back into a model's answer.

Reads the answer from a file or standard input and the replacements from the
``--token-map`` file ``wardcat scan`` wrote. The rules are the library's: a
placeholder that stood for more than one value is left as it is and reported,
a placeholder the map does not know is reported as foreign, and nothing is
guessed. ``--strict`` turns either into an error (exit 2) without printing.

Exit codes: ``0`` every placeholder was restored, ``1`` some were left in the
text (listed on standard error by type and reason, never by value), ``2`` a
usage error, an unreadable map, or ``--strict`` refusing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wardcat.cli._io import read_input, write_output
from wardcat.exceptions import ConfigError


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "restore",
        help="put the values from a --token-map back into a model's answer",
        allow_abbrev=False,
    )
    p.add_argument("file", nargs="?", default="-", help="the answer; - (default) for stdin")
    p.add_argument("--token-map", required=True, metavar="FILE", help="written by wardcat scan")
    p.add_argument("--output", "-o", metavar="PATH", help="write the restored text to a file")
    p.add_argument(
        "--strict", action="store_true", help="refuse (exit 2) instead of leaving placeholders"
    )
    p.add_argument("--sources", action="store_true", help="append the list of what was put back")
    p.set_defaults(run=run)


def _load(path: Path) -> list:
    from wardcat.core.models import Violation

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data["replacements"]
        return [
            Violation(
                entity_type=e["entity_type"],
                original=e["original"],
                start=0,
                end=0,
                action=e["action"],
                replacement=e["replacement"],
                confidence=float(e.get("confidence", 1.0)),
            )
            for e in entries
        ]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        raise ConfigError(f"{path.as_posix()} is not a wardcat token map") from None


def run(args: argparse.Namespace) -> int:
    from wardcat.cli import EXIT_CLEAN, EXIT_FOUND
    from wardcat.core.restore import restore_text
    from wardcat.exceptions import ContextMismatch

    violations = _load(Path(args.token_map))
    answer = read_input(args.file)
    try:
        restored = restore_text(answer, violations, strict=args.strict)
    except ContextMismatch as exc:
        raise ConfigError(f"refused: {exc}") from None
    if args.strict and not restored.is_complete:
        raise ConfigError("refused: some placeholders could not be restored")
    text = restored.with_sources() if args.sources else restored.text
    write_output(text, args.output)
    for item in restored.unrestored:
        if item.reason in ("ambiguous", "foreign"):
            print(
                f"warning: left a {item.entity_type} placeholder in place ({item.reason})",
                file=sys.stderr,
            )
    return EXIT_CLEAN if restored.is_complete else EXIT_FOUND

"""``wardcat scan``: sanitize one file or standard input."""

from __future__ import annotations

import argparse
import json
import sys

from wardcat.cli._guard_args import add_guard_args, build_guard
from wardcat.cli._io import read_input, write_output


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "scan",
        help="scan a file or stdin and print the sanitized text",
        allow_abbrev=False,
    )
    p.add_argument("file", nargs="?", default="-", help="path to read, or - for stdin")
    add_guard_args(p)
    out = p.add_argument_group("output")
    out.add_argument("--json", action="store_true", help="print the PII-free redacted() dict")
    out.add_argument("--output", "-o", metavar="PATH", help="write the result to a file")
    out.add_argument(
        "--quiet", "-q", action="store_true", help="print no warnings; the exit code still says it"
    )
    p.set_defaults(run=run)


def run(args: argparse.Namespace) -> int:
    from wardcat.cli import EXIT_CLEAN, EXIT_FOUND

    guard = build_guard(args)
    result = guard.scan(read_input(args.file))
    if args.json:
        write_output(json.dumps(result.redacted(), ensure_ascii=False), args.output)
    else:
        write_output(result.sanitized_text, args.output)
    if not args.quiet:
        for warning in result.warnings:
            print(f"warning: {warning}", file=sys.stderr)
    return EXIT_CLEAN if result.is_clean else EXIT_FOUND

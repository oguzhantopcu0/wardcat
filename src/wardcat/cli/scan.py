"""``wardcat scan``: sanitize standard input, files, or a JSON Lines dataset.

One input prints to standard output (or ``--output``). Several files need
``--output-dir``: each sanitized copy is written there under its own name.
``--jsonl`` reads one JSON object per line and replaces the ``--field`` value
with its sanitized text, keeping the other fields. Every text goes through one
``scan_batch`` call, so the NER layer sees them in a single pass.

The scan fails closed: if any text cannot be scanned (larger than the policy
allows, say), nothing is written and the command exits 2.

``--token-map FILE`` writes what ``wardcat restore`` needs to put the values
back into a model's answer. The file holds the original values — raw PII — so it
is only written when asked for, created readable by its owner alone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wardcat.cli._guard_args import add_guard_args, build_guard
from wardcat.cli._io import read_input, write_output
from wardcat.exceptions import ConfigError

if TYPE_CHECKING:  # pragma: no cover
    from wardcat import ScanResult

TOKEN_MAP_VERSION = 1


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "scan",
        help="sanitize stdin, files or a JSON Lines dataset",
        allow_abbrev=False,
    )
    p.add_argument(
        "files",
        nargs="*",
        default=["-"],
        metavar="FILE",
        help="paths to read; - (default) for stdin",
    )
    add_guard_args(p)
    out = p.add_argument_group("output")
    out.add_argument("--json", action="store_true", help="print the PII-free redacted() dict")
    out.add_argument("--output", "-o", metavar="PATH", help="write the result to a file")
    out.add_argument("--output-dir", metavar="DIR", help="with several files: write each copy here")
    out.add_argument(
        "--jsonl", action="store_true", help="the input is JSON Lines; sanitize one field per line"
    )
    out.add_argument("--field", default="text", help="the field --jsonl sanitizes (default: text)")
    out.add_argument(
        "--token-map",
        metavar="FILE",
        help="write the values needed by `wardcat restore` (raw PII; owner-only file)",
    )
    out.add_argument(
        "--quiet", "-q", action="store_true", help="print no warnings; the exit code still says it"
    )
    p.set_defaults(run=run)


def run(args: argparse.Namespace) -> int:
    from wardcat.cli import EXIT_CLEAN, EXIT_FOUND

    _check_args(args)
    guard = build_guard(args)
    if args.jsonl:
        results = _scan_jsonl(guard, args)
    elif len(args.files) > 1:
        results = _scan_files(guard, args)
    else:
        results = _scan_one(guard, args)
    if args.token_map:
        write_token_map(Path(args.token_map), results)
    if not args.quiet:
        for warning in dict.fromkeys(w for r in results for w in r.warnings):
            print(f"warning: {warning}", file=sys.stderr)
    return EXIT_CLEAN if all(r.is_clean for r in results) else EXIT_FOUND


def _check_args(args: argparse.Namespace) -> None:
    if args.jsonl and len(args.files) > 1:
        raise ConfigError("--jsonl reads one input")
    if len(args.files) > 1 and not args.output_dir:
        raise ConfigError("several files need --output-dir")
    if args.output_dir and args.output:
        raise ConfigError("--output and --output-dir cannot be combined")
    if args.output_dir and "-" in args.files:
        raise ConfigError("stdin (-) cannot be combined with --output-dir")


def _batch(guard: Any, texts: list[str], names: list[str]) -> list[ScanResult]:
    results = guard.scan_batch(texts)
    failed = [name for name, r in zip(names, results, strict=True) if r.scan_error]
    if failed:
        # scan_batch hands back the original text for an item it could not scan;
        # writing that out would publish it unmasked.
        raise ConfigError(f"could not scan {', '.join(failed)}; nothing was written")
    return list(results)


def _scan_one(guard: Any, args: argparse.Namespace) -> list[ScanResult]:
    name = args.files[0]
    result = guard.scan(read_input(name))
    if args.json:
        write_output(json.dumps(result.redacted(), ensure_ascii=False), args.output)
    else:
        write_output(result.sanitized_text, args.output)
    return [result]


def _scan_files(guard: Any, args: argparse.Namespace) -> list[ScanResult]:
    paths = [Path(f) for f in args.files]
    names = [p.name for p in paths]
    duplicate = sorted({n for n in names if names.count(n) > 1})
    if duplicate:
        raise ConfigError(f"several inputs share a name: {', '.join(duplicate)}")
    texts = [read_input(str(p)) for p in paths]
    results = _batch(guard, texts, [p.as_posix() for p in paths])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, result in zip(names, results, strict=True):
        body = (
            json.dumps(result.redacted(), ensure_ascii=False)
            if args.json
            else result.sanitized_text
        )
        write_output(body, str(out_dir / (name + ".json" if args.json else name)))
    return results


def _scan_jsonl(guard: Any, args: argparse.Namespace) -> list[ScanResult]:
    records: list[dict[str, Any]] = []
    for number, line in enumerate(read_input(args.files[0]).splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            raise ConfigError(f"line {number} is not JSON") from None
        if not isinstance(record, dict) or not isinstance(record.get(args.field), str):
            raise ConfigError(f"line {number} has no string field {args.field!r}")
        records.append(record)
    texts = [r[args.field] for r in records]
    results = _batch(guard, texts, [f"line {i}" for i in range(1, len(texts) + 1)])
    lines = []
    for record, result in zip(records, results, strict=True):
        if args.json:
            lines.append(json.dumps(result.redacted(), ensure_ascii=False))
        else:
            lines.append(
                json.dumps({**record, args.field: result.sanitized_text}, ensure_ascii=False)
            )
    write_output("\n".join(lines), args.output)
    return results


def write_token_map(path: Path, results: list[ScanResult]) -> None:
    """Record every replacement that can be put back, in an owner-only file."""
    entries = [
        {
            "replacement": v.replacement,
            "original": v.original,
            "entity_type": v.entity_type,
            "action": getattr(v.action, "value", v.action),
            "confidence": v.confidence,
        }
        for r in results
        for v in r.violations
        if v.replacement is not None
    ]
    payload = json.dumps(
        {
            "version": TOKEN_MAP_VERSION,
            "warning": "raw PII: keep this file private and delete it after use",
            "context_ids": sorted({r.context_id for r in results if r.context_id}),
            "replacements": entries,
        },
        ensure_ascii=False,
        indent=2,
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(payload + "\n")
    if os.name == "posix":
        os.chmod(path, 0o600)  # an existing file keeps its old mode through O_CREAT

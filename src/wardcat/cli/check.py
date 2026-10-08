"""``wardcat check``: find secrets and PII in files, for pre-commit and CI.

It walks paths, reports each finding as ``file:line:col ENTITY`` and never
rewrites anything. Every way a file can go unscanned is loud: a file that
cannot be read is an error (exit 2), a binary file is listed as skipped, and a
document too large for one scan is split on line breaks under the policy's own
``max_text_bytes`` — a single line longer than that is an error rather than a
cut through a value. Output names entity types and positions, never the values.

``--git-diff`` scans only the lines a change adds: the staged changes by
default (what the next commit will hold), or ``git diff REF`` with a ref or
range. Findings carry the line numbers of the new file, so a baseline written
from a full ``check`` applies. Paths given with it narrow the diff.

Exit codes: ``0`` nothing found, ``1`` findings, ``2`` usage, config or read
error, ``3`` a degraded scan under ``--strict``.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from wardcat.cli._guard_args import add_guard_args, build_guard
from wardcat.cli._io import read_input, write_output
from wardcat.exceptions import ConfigError

if TYPE_CHECKING:  # pragma: no cover
    from wardcat import Wardcat

# Never worth scanning, and enough to flood a report.
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".tox",
        "dist",
        "build",
        ".idea",
        ".vscode",
    }
)
# NER-only types: without a model they are never found, which an explicit
# request deserves to hear about.
_MODEL_ONLY = frozenset({"PERSON", "ORG", "LOCATION", "NRP"})
_BOMS = (
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
    (b"\xef\xbb\xbf", "utf-8-sig"),
)
# Headroom under the byte limit for the scan's own bookkeeping.
_CHUNK_MARGIN = 1024


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    col: int
    entity_type: str
    action: str
    confidence: float

    @property
    def fingerprint(self) -> str:
        """Suppression key for ``--baseline``: file, line and type — never the value."""
        return f"{self.file}:{self.line}:{self.entity_type}"


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "check",
        help="find secrets and PII in files; exit 1 if any (pre-commit, CI)",
        allow_abbrev=False,
    )
    p.add_argument("paths", nargs="*", metavar="PATH", help="files and directories; - for stdin")
    p.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="descend into directories (default: yes)",
    )
    add_guard_args(p, default_action="warn")
    files = p.add_argument_group("files")
    files.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GLOB",
        help="only scan files matching; repeatable",
    )
    files.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="skip files matching; repeatable",
    )
    files.add_argument(
        "--git-diff",
        nargs="?",
        const="",
        default=None,
        metavar="REF",
        help="scan only added lines: staged changes, or `git diff REF` (e.g. origin/main...HEAD)",
    )
    files.add_argument(
        "--stdin-filename",
        default="<stdin>",
        metavar="NAME",
        help="name to report for content read from -",
    )
    files.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        metavar="N",
        help="scan in N processes; worth it with --ner or --llm (default: 1)",
    )
    out = p.add_argument_group("output")
    out.add_argument("--format", choices=["text", "jsonl", "sarif"], default="text")
    out.add_argument("--output", "-o", metavar="PATH", help="write the report to a file")
    out.add_argument(
        "--baseline",
        metavar="FILE",
        help="suppress findings recorded in this file (see --write-baseline)",
    )
    out.add_argument(
        "--write-baseline",
        action="store_true",
        help="record the current findings in --baseline and exit 0",
    )
    out.add_argument(
        "--quiet", "-q", action="store_true", help="print nothing; only set the exit code"
    )
    p.set_defaults(run=run)


def run(args: argparse.Namespace) -> int:
    from wardcat.cli import EXIT_CLEAN, EXIT_FOUND

    if args.write_baseline and not args.baseline:
        raise ConfigError("--write-baseline needs --baseline FILE")
    if args.jobs < 1:
        raise ConfigError("--jobs must be at least 1")
    if args.git_diff is None and not args.paths:
        raise ConfigError("give a PATH to check, - for stdin, or --git-diff")
    if args.git_diff is not None and "-" in args.paths:
        raise ConfigError("--git-diff cannot be combined with stdin (-)")
    guard = _guard(args)

    findings: list[Finding] = []
    skipped: list[str] = []
    warnings: dict[str, None] = {}
    line_maps: dict[str, list[int]] = {}
    inputs = (
        _diff_inputs(args, skipped, line_maps)
        if args.git_diff is not None
        else _inputs(args, skipped)
    )
    if args.jobs == 1:
        scanned: Iterable[tuple[list[Finding], list[str]]] = (
            scan_text(guard, text, name) for name, text in inputs
        )
    else:
        scanned = _scan_in_processes(args, list(inputs))
    for found, notes in scanned:
        findings.extend(found)
        warnings.update(dict.fromkeys(notes))
    if line_maps:
        findings = [_in_file(f, line_maps[f.file]) for f in findings]

    if args.write_baseline:
        count = write_baseline(Path(args.baseline), findings)
        _say(args, f"wrote {count} fingerprint(s) to {args.baseline}")
        return EXIT_CLEAN
    if args.baseline:
        allowed = load_baseline(Path(args.baseline))
        findings = [f for f in findings if f.fingerprint not in allowed]

    if not args.quiet:
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)
        if skipped:
            print(f"skipped {len(skipped)} binary file(s): {', '.join(skipped)}", file=sys.stderr)
        report = FORMATTERS[args.format](findings)
        if report or args.output:
            write_output(report, args.output)
        if args.format == "text":
            files = len({f.file for f in findings})
            print(
                f"{len(findings)} finding(s) in {files} file(s)" if findings else "nothing found",
                file=sys.stderr,
            )
    return EXIT_FOUND if findings else EXIT_CLEAN


def _guard(args: argparse.Namespace) -> Wardcat:
    explicit = bool(args.config or args.preset or args.entity or args.group)
    if not explicit:
        args.group = ["all"]
    guard = build_guard(args, scope_required=False)
    policy = guard.entity_policy()
    if not policy:
        raise ConfigError("nothing to scan for: the policy enables no entity type")
    if explicit and not (args.ner or args.ner_language or args.llm):
        missing = sorted(_MODEL_ONLY & set(policy))
        if missing:
            _say(args, f"note: {', '.join(missing)} need --ner or --llm and are not checked")
    return guard


_WORKER_GUARD: Wardcat | None = None


def _start_worker(args: argparse.Namespace) -> None:
    global _WORKER_GUARD
    _WORKER_GUARD = build_guard(args, scope_required=False, warn=False)


def _scan_in_worker(item: tuple[str, str]) -> tuple[list[Finding], list[str]]:
    name, text = item
    assert _WORKER_GUARD is not None
    return scan_text(_WORKER_GUARD, text, name)


def _scan_in_processes(
    args: argparse.Namespace, items: list[tuple[str, str]]
) -> Iterator[tuple[list[Finding], list[str]]]:
    """Each process builds its own guard (and loads its own model) once."""
    from concurrent.futures import ProcessPoolExecutor

    worker_args = argparse.Namespace(**{k: v for k, v in vars(args).items() if k != "run"})
    with ProcessPoolExecutor(
        max_workers=min(args.jobs, max(1, len(items))),
        initializer=_start_worker,
        initargs=(worker_args,),
    ) as pool:
        yield from pool.map(_scan_in_worker, items)


def _say(args: argparse.Namespace, message: str) -> None:
    if not args.quiet:
        print(message, file=sys.stderr)


def _inputs(args: argparse.Namespace, skipped: list[str]) -> Iterator[tuple[str, str]]:
    if "-" in args.paths:
        yield args.stdin_filename, read_input("-")
    paths = [Path(p) for p in args.paths if p != "-"]
    for path in iter_files(
        paths, recursive=args.recursive, include=args.include, exclude=args.exclude
    ):
        text = read_text_file(path)
        if text is None:
            skipped.append(path.as_posix())
            continue
        yield path.as_posix(), text


_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _diff_inputs(
    args: argparse.Namespace, skipped: list[str], line_maps: dict[str, list[int]]
) -> Iterator[tuple[str, str]]:
    """Each changed file's added lines, joined; *line_maps* records their real numbers."""

    def matches(name: str, patterns: list[str]) -> bool:
        return any(
            fnmatch.fnmatch(name, g) or fnmatch.fnmatch(Path(name).name, g) for g in patterns
        )

    for name, numbers, lines in parse_added_lines(_git_diff(args)):
        if matches(name, args.exclude) or (args.include and not matches(name, args.include)):
            continue
        if lines is None:
            skipped.append(name)
            continue
        line_maps[name] = numbers
        yield name, "".join(lines)


def _git_diff(args: argparse.Namespace) -> str:
    command = [
        "git",
        "-c",
        "core.quotepath=off",
        "diff",
        "--no-color",
        "--no-ext-diff",
        "--no-textconv",
        "--unified=0",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        "--diff-filter=d",
    ]
    command += ["--cached"] if args.git_diff == "" else [args.git_diff]
    command += ["--", *args.paths]
    try:
        inside = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, check=False
        )
        if inside.returncode != 0:
            raise ConfigError("--git-diff must run inside a git work tree")
        done = subprocess.run(command, capture_output=True, check=False)
    except OSError:
        raise ConfigError("--git-diff needs git on the PATH") from None
    if done.returncode != 0:
        reason = done.stderr.decode("utf-8", errors="replace").strip().splitlines()
        raise ConfigError(f"git diff failed: {reason[0] if reason else done.returncode}")
    return done.stdout.decode("utf-8", errors="replace")


def parse_added_lines(diff: str) -> Iterator[tuple[str, list[int], list[str] | None]]:
    """``(path, line numbers, added lines)`` per file of a ``git diff -U0``.

    The lines keep their line breaks. A binary file comes back with ``None``
    for its lines, so the caller can say it was not scanned. Inside a hunk the
    header's count decides what is content, so an added line that itself starts
    with ``++ `` is not taken for a file header.
    """
    name: str | None = None
    numbers: list[int] = []
    lines: list[str] = []
    binary = False
    number = remaining = 0

    def flush() -> Iterator[tuple[str, list[int], list[str] | None]]:
        if name is not None and (lines or binary):
            yield name, numbers, None if binary else lines

    for line in (piece + "\n" for piece in diff.split("\n")[:-1]):
        if remaining:
            if line.startswith("+"):
                numbers.append(number)
                lines.append(line[1:])
                number += 1
                remaining -= 1
            continue  # a removed line, or git's "no newline at end of file" note
        if line.startswith("diff --git "):
            yield from flush()
            name, numbers, lines, binary = None, [], [], False
            head = _unquote(line[len("diff --git ") :].rstrip("\n"))
            if head.startswith("a/") and " b/" in head:
                name = head.split(" b/", 1)[1]  # until +++ names it exactly
        elif line.startswith("+++ "):
            target = _unquote(line[4:].rstrip("\n"))
            name = target[2:] if target.startswith("b/") else name
        elif line.startswith("Binary files "):
            binary = True
        elif match := _HUNK.match(line):
            number = int(match.group(1))
            remaining = int(match.group(2)) if match.group(2) is not None else 1
    yield from flush()


def _unquote(path: str) -> str:
    """Undo git's C-style quoting of a path with special characters."""
    if not (path.startswith('"') and path.endswith('"')):
        return path
    raw = bytearray()
    body = path[1:-1]
    i = 0
    escapes = {"n": 10, "t": 9, '"': 34, "\\": 92, "a": 7, "b": 8, "f": 12, "r": 13, "v": 11}
    while i < len(body):
        char = body[i]
        if char == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            if nxt in "01234567" and i + 3 < len(body) + 1:
                raw.append(int(body[i + 1 : i + 4], 8))
                i += 4
                continue
            raw.append(escapes.get(nxt, ord(nxt)))
            i += 2
            continue
        raw += char.encode("utf-8")
        i += 1
    return raw.decode("utf-8", errors="replace")


def _in_file(finding: Finding, numbers: list[int]) -> Finding:
    """Move a finding from its place among the added lines to its line in the file."""
    from dataclasses import replace

    return replace(finding, line=numbers[finding.line - 1])


def iter_files(
    paths: Iterable[Path],
    *,
    recursive: bool,
    include: list[str],
    exclude: list[str],
) -> Iterator[Path]:
    """The files to scan. A path that does not exist is an error, not a skip."""

    def matches(f: Path, patterns: list[str]) -> bool:
        return any(fnmatch.fnmatch(f.name, g) or fnmatch.fnmatch(f.as_posix(), g) for g in patterns)

    def wanted(f: Path) -> bool:
        if matches(f, exclude):
            return False
        return not include or matches(f, include)

    seen: set[Path] = set()
    for p in paths:
        if p.is_dir():
            candidates: Iterable[Path] = sorted(p.rglob("*") if recursive else p.glob("*"))
            candidates = (f for f in candidates if f.is_file() and not (set(f.parts) & _SKIP_DIRS))
        elif p.is_file():
            candidates = [p]
        else:
            raise ConfigError(f"no such file or directory: {p.as_posix()}")
        for f in candidates:
            resolved = f.resolve()
            if resolved not in seen and wanted(f):
                seen.add(resolved)
                yield f


def read_text_file(path: Path) -> str | None:
    """The file's text; ``None`` for a binary file. A read error raises (exit 2).

    A byte-order mark decides the encoding (Windows tools write UTF-16 with one).
    Without one the file is read as UTF-8, with undecodable bytes replaced rather
    than the file skipped: a card number in a Latin-1 file is still a card number.
    """
    data = path.read_bytes()
    for bom, encoding in _BOMS:
        if data.startswith(bom):
            return data.decode(encoding, errors="replace")
    if b"\x00" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


def scan_text(guard: Wardcat, text: str, name: str) -> tuple[list[Finding], list[str]]:
    """Findings and warnings for one document, scanned in line-aligned pieces."""
    findings: list[Finding] = []
    warnings: list[str] = []
    for offset, piece in _pieces(text, guard.max_text_bytes - _CHUNK_MARGIN, name):
        result = guard.scan(piece)
        warnings.extend(result.warnings)
        for v in result.violations:
            line, col = _line_col(text, offset + v.start)
            action = getattr(v.action, "value", v.action)
            findings.append(Finding(name, line, col, v.entity_type, str(action), v.confidence))
    return findings, warnings


def _pieces(text: str, budget: int, name: str) -> Iterator[tuple[int, str]]:
    """``(character offset, piece)`` pairs, each under *budget* UTF-8 bytes, cut at line ends."""
    if len(text.encode("utf-8")) <= budget:
        yield 0, text
        return
    start, size = 0, 0
    piece: list[str] = []
    position = 0
    for line in text.splitlines(keepends=True):
        width = len(line.encode("utf-8"))
        if width > budget:
            number = text.count("\n", 0, position) + 1
            raise ConfigError(
                f"{name}: line {number} is longer than the scan limit "
                f"({budget:,} bytes); raise max_text_bytes in the policy to check it"
            )
        if size + width > budget:
            yield start, "".join(piece)
            start, piece, size = position, [], 0
        piece.append(line)
        size += width
        position += len(line)
    if piece:
        yield start, "".join(piece)


def _line_col(text: str, offset: int) -> tuple[int, int]:
    """1-based line and column of a character offset."""
    line = text.count("\n", 0, offset) + 1
    return line, offset - text.rfind("\n", 0, offset)


def load_baseline(path: Path) -> set[str]:
    """Fingerprints recorded in a baseline. A missing file suppresses nothing."""
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path.as_posix()} is not a baseline file: {exc}") from None
    prints = data.get("fingerprints") if isinstance(data, dict) else None
    if not isinstance(prints, list):
        raise ConfigError(f"{path.as_posix()} is not a baseline file: no 'fingerprints' list")
    return {str(x) for x in prints}


def write_baseline(path: Path, findings: list[Finding]) -> int:
    prints = sorted({f.fingerprint for f in findings})
    path.write_text(
        json.dumps({"version": 1, "fingerprints": prints}, indent=2) + "\n", encoding="utf-8"
    )
    return len(prints)


def format_text(findings: list[Finding]) -> str:
    return "\n".join(
        f"{f.file}:{f.line}:{f.col}  {f.entity_type}  ({f.confidence:.2f})" for f in findings
    )


def format_jsonl(findings: list[Finding]) -> str:
    """One JSON object per finding: file, line, col, entity_type, action, confidence."""
    return "\n".join(
        json.dumps(
            {
                "file": f.file,
                "line": f.line,
                "col": f.col,
                "entity_type": f.entity_type,
                "action": f.action,
                "confidence": f.confidence,
            },
            ensure_ascii=False,
        )
        for f in findings
    )


def format_sarif(findings: list[Finding]) -> str:
    """SARIF 2.1.0, for GitHub code scanning (upload-sarif)."""
    from wardcat import __version__

    results = [
        {
            "ruleId": f.entity_type,
            "level": "error",
            "message": {"text": f"Possible {f.entity_type} detected"},
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {"uri": f.file},
                        "region": {"startLine": f.line, "startColumn": f.col},
                    }
                }
            ],
        }
        for f in findings
    ]
    document = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "wardcat",
                        "version": __version__,
                        "informationUri": "https://github.com/oguzhantopcu0/wardcat",
                        "rules": [
                            {"id": r, "name": r} for r in sorted({f.entity_type for f in findings})
                        ],
                    }
                },
                "results": results,
            }
        ],
    }
    return json.dumps(document, ensure_ascii=False, indent=2)


FORMATTERS = {"text": format_text, "jsonl": format_jsonl, "sarif": format_sarif}

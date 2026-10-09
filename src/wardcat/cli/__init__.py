"""The ``wardcat`` command.

::

    wardcat scan     [FILE...|-] [guard options] [--json] [--output PATH|--output-dir DIR]
                     [--jsonl --field NAME] [--token-map FILE]
    wardcat restore  [FILE|-] --token-map FILE [--strict] [--sources]
    wardcat check    PATH... [guard options] [--format text|jsonl|sarif] [--baseline FILE]
    wardcat is-sensitive [FILE|-] --llm MODEL [LLM options]
    wardcat serve    [guard options] [--host 127.0.0.1] [--port 8787]   (wardcat[serve])
    wardcat check-config policy.yaml
    wardcat entities [--layer regex|ner|llm]
    wardcat presets  [NAME]
    wardcat models   list | pull MODEL
    wardcat          [--resume ID|--continue] [--salt-env VAR]   (a terminal: the interactive screen)
    wardcat sessions [--delete ID]
    wardcat --version

Exit codes: ``0`` clean, ``1`` something was found (or the text is sensitive),
``2`` a configuration, usage or input error, ``3`` the scan could not be
completed — degraded under ``--strict``, or the LLM backend unreachable for
``is-sensitive``.

The salt and API keys are never taken on the command line — a process list
and a shell history would keep them; ``--salt-env`` and ``--llm-api-key-env``
name the environment variables to read. Output never carries a value that was
found.

This package is not public API: import :func:`main` and the ``EXIT_*`` codes,
nothing else. Every subcommand imports what it needs when it runs, so
``import wardcat`` never loads it and ``wardcat --help`` loads no detector.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from wardcat.exceptions import ConfigError, DegradedScanError, WardcatError

EXIT_CLEAN, EXIT_FOUND, EXIT_CONFIG, EXIT_DEGRADED = 0, 1, 2, 3


class BackendUnavailable(Exception):
    """A layer the command depends on could not be reached (exit 3)."""


def _version() -> str:
    from wardcat import __version__

    return f"wardcat {__version__}"


def _parser() -> argparse.ArgumentParser:
    from wardcat.cli import (
        _sessions,
        check,
        check_config,
        completion,
        entities,
        hook,
        is_sensitive,
        models,
        presets,
        restore,
        scan,
        serve,
        shell,
    )

    parser = argparse.ArgumentParser(
        prog="wardcat",
        description="PII detection and anonymization.",
        allow_abbrev=False,
    )
    parser.add_argument("--version", "-V", action="version", version=_version())
    shell.add_arguments(parser)
    sub = parser.add_subparsers(dest="command")
    for module in (
        scan,
        restore,
        check,
        is_sensitive,
        serve,
        check_config,
        entities,
        presets,
        models,
        hook,
        completion,
        _sessions,
    ):
        module.register(sub)
    return parser


def _warn_about_the_old_cli() -> None:
    """The retired wardcat-cli package installed a ``wardcat`` command too."""
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        distribution("wardcat-cli")
    except PackageNotFoundError:
        return
    print(
        "warning: the old wardcat-cli package is installed and also provides a "
        "`wardcat` command; its features are part of wardcat now. Remove it with: "
        "pip uninstall wardcat-cli",
        file=sys.stderr,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _warn_about_the_old_cli()
    screen = args.screen_resume or args.screen_continue or args.screen_salt_env
    try:
        if args.command is None:
            from wardcat.cli import shell

            return shell.run(args, parser)
        if screen:
            raise ConfigError(
                "--resume, --continue and --salt-env before a command open the screen"
            )
        return int(args.run(args))
    except DegradedScanError as exc:
        for warning in exc.warnings:
            print(f"error: {warning}", file=sys.stderr)
        return EXIT_DEGRADED
    except BackendUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DEGRADED
    except (ConfigError, WardcatError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG


__all__ = ["EXIT_CLEAN", "EXIT_CONFIG", "EXIT_DEGRADED", "EXIT_FOUND", "main"]

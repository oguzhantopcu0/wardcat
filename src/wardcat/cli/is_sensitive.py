"""``wardcat is-sensitive``: ask the LLM layer whether a text is sensitive at all.

Prints ``sensitive`` or ``clean``. Exit codes: ``0`` clean, ``1`` sensitive,
``2`` a usage, config or read error (an input over the size limit included —
it is refused, not split, because a verdict on part of a text is not a verdict
on the text), ``3`` the backend could not be reached or its circuit is open.
The check fails closed: a backend that cannot answer never yields ``clean``.
"""

from __future__ import annotations

import argparse
import json

from wardcat.cli._guard_args import add_llm_args, llm_kwargs
from wardcat.cli._io import read_input


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "is-sensitive",
        help="ask the LLM layer whether a file or stdin is sensitive",
        allow_abbrev=False,
    )
    p.add_argument("file", nargs="?", default="-", help="path to read, or - for stdin")
    p.add_argument("--llm", metavar="MODEL", required=True, help="the model to ask")
    add_llm_args(p)
    p.add_argument(
        "--llm-language",
        metavar="LANG",
        help="ask in this language (tr, de, fr); others use the multilingual English prompt",
    )
    p.add_argument(
        "--categories",
        action="store_true",
        help="also name the kinds of sensitive content (pii, credentials, health, ...) as JSON",
    )
    p.set_defaults(run=run)


def run(args: argparse.Namespace) -> int:
    import httpx

    from wardcat import Wardcat
    from wardcat.cli import EXIT_CLEAN, EXIT_FOUND, BackendUnavailable
    from wardcat.llm.circuit import CircuitOpen

    text = read_input(args.file)  # a read error is exit 2, before any backend call
    kwargs = llm_kwargs(args)
    if args.llm_language:
        kwargs["language"] = args.llm_language
    guard = Wardcat().with_llm(model=args.llm, **kwargs)  # type: ignore[arg-type]
    try:
        if args.categories:
            verdict = guard.classify(text)
            sensitive = verdict.sensitive
        else:
            sensitive = guard.is_sensitive(text)
    except (ConnectionError, TimeoutError, httpx.HTTPError, CircuitOpen) as exc:
        raise BackendUnavailable(
            f"the LLM backend could not answer ({type(exc).__name__}); not judged"
        ) from None
    if args.categories:
        # The model's one-line reason may quote the text, so it is not printed.
        print(json.dumps({"sensitive": sensitive, "categories": list(verdict.categories)}))
    else:
        print("sensitive" if sensitive else "clean")
    return EXIT_FOUND if sensitive else EXIT_CLEAN

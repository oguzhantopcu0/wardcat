"""``wardcat models``: the SpaCy models the NER layer can use, and installing one.

The other commands never download a model; this is the explicit way to get one.
``pull`` only installs a model from wardcat's catalog, so a typo cannot fetch an
arbitrary package.
"""

from __future__ import annotations

import argparse
import sys


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "models", help="list or install the SpaCy models for NER", allow_abbrev=False
    )
    actions = p.add_subparsers(dest="models_command", required=True)
    listing = actions.add_parser(
        "list", help="the catalog, with what is installed", allow_abbrev=False
    )
    listing.add_argument("--language", metavar="LANG", help="only this language, e.g. tr")
    listing.set_defaults(run=run_list)
    pull = actions.add_parser("pull", help="install a model from the catalog", allow_abbrev=False)
    pull.add_argument("name", help="a catalog model, e.g. tr_core_news_md")
    pull.set_defaults(run=run_pull)


def run_list(args: argparse.Namespace) -> int:
    import warnings

    from wardcat.cli import EXIT_CLEAN
    from wardcat.ner.downloader import is_installed
    from wardcat.ner.spacy_catalog import SPACY_CATALOG

    rows = [m for m in SPACY_CATALOG if not args.language or m.lang_code == args.language.lower()]
    with warnings.catch_warnings():
        # SpaCy warns about each installed model's version pin while listing them.
        warnings.simplefilter("ignore")
        installed = {m.name for m in rows if is_installed(m.name)}
    width = max((len(m.name) for m in rows), default=4)
    for m in rows:
        state = "installed" if m.name in installed else "-"
        notes = []
        if m.recommended:
            notes.append("recommended")
        if not m.has_ner:
            notes.append("no NER component")
        if m.incompatible:
            notes.append("incompatible with this SpaCy")
        print(
            f"{m.name:<{width}}  {m.lang_code}  {m.size:<3}  ~{m.ram_mb:>4} MB  {state:<9}  {', '.join(notes)}"
        )
    return EXIT_CLEAN


def run_pull(args: argparse.Namespace) -> int:
    from wardcat.cli import EXIT_CLEAN
    from wardcat.exceptions import ConfigError
    from wardcat.ner.spacy_catalog import get_spacy_model

    info = get_spacy_model(args.name)
    if info is None:
        raise ConfigError(f"{args.name!r} is not in the catalog; see `wardcat models list`")
    if not info.has_ner:
        raise ConfigError(f"{args.name} has no NER component and cannot be used by the NER layer")
    try:
        from wardcat.ner.downloader import download_model, is_installed
    except ImportError:
        raise ConfigError("the NER layer needs SpaCy: pip install 'wardcat[ner]'") from None
    if is_installed(args.name):
        print(f"{args.name} is already installed", file=sys.stderr)
        return EXIT_CLEAN
    try:
        download_model(args.name, verbose=True)
    except ImportError:
        raise ConfigError("the NER layer needs SpaCy: pip install 'wardcat[ner]'") from None
    print(f"installed {args.name}", file=sys.stderr)
    return EXIT_CLEAN

"""Reading input and writing output as UTF-8, on every platform.

Windows hands a console or pipe to Python in the ANSI code page (cp1252 on a
Western install), which has no ş, ğ or İ: a Turkish document piped through
``wardcat scan`` came back with question marks. A real stream is switched to
UTF-8 here; anything else (a ``StringIO`` in a test, an embedding application's
own stream) is used as it is.
"""

from __future__ import annotations

import io
import sys
from typing import TextIO

from wardcat.exceptions import ConfigError


def _utf8(stream: TextIO | None, *, reading: bool) -> TextIO:
    if stream is None:
        raise ConfigError(
            "no standard input is attached; pass a file path instead"
            if reading
            else "no standard output is attached"
        )
    if isinstance(stream, io.TextIOWrapper):
        # utf-8-sig on input also drops the BOM Windows tools put in front.
        stream.reconfigure(encoding="utf-8-sig" if reading else "utf-8")
    return stream


def read_input(path: str) -> str:
    """The text at *path*, or standard input for ``-``."""
    if path == "-":
        stdin = _utf8(sys.stdin, reading=True)
        if stdin.isatty():
            print(
                "reading standard input; end it with Ctrl-D (Ctrl-Z, Enter on Windows)",
                file=sys.stderr,
            )
        return stdin.read()
    with open(path, encoding="utf-8-sig") as fh:
        return fh.read()


def write_output(text: str, path: str | None = None) -> None:
    """Write *text* with a trailing newline, to *path* or standard output."""
    if not text.endswith("\n"):
        text += "\n"
    if path:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        return
    _utf8(sys.stdout, reading=False).write(text)

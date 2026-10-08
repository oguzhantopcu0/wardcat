"""What the interactive screen draws: banner, tables, help, the ``/`` palette.

Plain ANSI, no rendering library. Colour is used only on a terminal and never
when ``NO_COLOR`` is set.
"""

from __future__ import annotations

import os
import shutil
import sys

BANNER = r"""
 ▓▓▒░          ░▒▓▓
▒█ ▒▓▓▒▒▒▒▒▒▒▒▓▓▒ █▒
░█   ▒▒▒░░░░▒▒▒   █░   ██╗    ██╗ █████╗ ██████╗ ██████╗  ██████╗ █████╗ ████████╗
░█▒              ▒█░   ██║    ██║██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗╚══██╔══╝
█▒  ▓▓▒      ▒▓▓  ▒█   ██║ █╗ ██║███████║██████╔╝██║  ██║██║     ███████║   ██║
▓▓   ▒▓▒    ▒▓▒   ▓▓   ██║███╗██║██╔══██║██╔══██╗██║  ██║██║     ██╔══██║   ██║
 █▓              ▓█    ╚███╔███╔╝██║  ██║██║  ██║██████╔╝╚██████╗██║  ██║   ██║
  █▓            ▓█      ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝  ╚═════╝╚═╝  ╚═╝   ╚═╝
   ▓█▓        ▓█▓
     ▓▓▓▒  ▒▓▓▓
       ▒▓▓▓▓▒
"""

_CODES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "cyan": "36"}


def colour_on() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def paint(text: str, *styles: str) -> str:
    if not styles or not colour_on():
        return text
    codes = ";".join(_CODES[s] for s in styles)
    return f"\x1b[{codes}m{text}\x1b[0m"


def banner(version: str) -> str:
    art = BANNER.strip("\n")
    width = max(len(line) for line in art.splitlines())
    if shutil.get_terminal_size((80, 24)).columns < width + 2:
        art = "\n".join(line[:20].rstrip() for line in art.splitlines())  # the cat alone
        width = 20
    lines = [paint(line, "bold", "cyan") for line in art.splitlines()]
    lines.append(paint(f"wardcat {version}".rjust(width), "dim"))
    return "\n".join(lines)


def table(headers: list[str], rows: list[list[str]]) -> str:
    """Columns padded to their widest cell; the last one wraps at the terminal's edge."""
    import textwrap

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    room = shutil.get_terminal_size((100, 24)).columns - sum(w + 2 for w in widths[:-1]) - 1
    widths[-1] = min(widths[-1], max(room, 20))  # never squeezed below 20 to wrap
    indent = " " * sum(w + 2 for w in widths[:-1])

    def line(cells: list[str]) -> str:
        head = "  ".join(c.ljust(w) for c, w in zip(cells[:-1], widths, strict=False))
        wrapped = textwrap.wrap(cells[-1], widths[-1]) or [""]
        first = (head + "  " if head else "") + wrapped[0]
        return "\n".join([first.rstrip(), *(indent + more for more in wrapped[1:])])

    out = [paint(line(headers), "bold"), paint("  ".join("─" * w for w in widths), "dim")]
    out.extend(line(r) for r in rows)
    return "\n".join(out)


HELP = [
    ("Scan", ""),
    ("scan [TEXT]", "sanitize TEXT and list what was found; asks for it when omitted"),
    ("Filters", "which entity types are looked for, and the action on each"),
    ("add filter ENTITY [--action A]", "look for ENTITY; A is redact (default), mask, hash, ..."),
    ("remove filter ENTITY", "stop looking for it"),
    ("preset NAME", "replace the filters with a preset's (kvkk, gdpr, pci_dss, ...)"),
    ("filters", "the active filters and their actions"),
    ("list filters [--active|--inactive]", "every type wardcat knows, and which layer finds it"),
    ("Layers", "the detectors that do the finding"),
    ("add layer regex", "patterns and checksums: cards, IBAN, TC, e-mail, keys"),
    ("add layer ner tr [SIZE]", "names, organisations, places, with a SpaCy model"),
    ("add layer llm", "a local LLM; asks for backend, model and address"),
    ("remove layer LAYER", "switch one off (one must stay on)"),
    ("layers", "the active layers"),
    ("Service", ""),
    ("serve [--port P]", "serve this session's policy on 127.0.0.1 in the background"),
    ("stop-serve", "stop it"),
    ("Screen", ""),
    ("clear", "clear the screen"),
    ("help", "this list"),
    ("quit", "save the session and leave (or Ctrl-D)"),
]

ACTIONS_NOTE = (
    "actions: redact [EMAIL] · mask ****1111 · hash [EMAIL:3f2a…] · tokenize · "
    "surrogate · warn (kept, only reported)"
)


def help_text() -> str:
    width = max(len(cmd) for cmd, _ in HELP)
    out = []
    for cmd, desc in HELP:
        if cmd[0].isupper():
            out.append("")
            out.append(paint(cmd, "bold") + (f"  {paint(desc, 'dim')}" if desc else ""))
        else:
            out.append(f"  {paint(cmd.ljust(width), 'cyan')}  {desc}")
    out.append("")
    out.append(paint(ACTIONS_NOTE, "dim"))
    return "\n".join(out)


COMMANDS = [
    "scan",
    "add",
    "remove",
    "preset",
    "filters",
    "list",
    "layers",
    "serve",
    "stop-serve",
    "clear",
    "help",
    "quit",
    "exit",
]

# The palette `/` opens: (command, what it does), filtered as you type.
PALETTE = [
    ("scan", "sanitize a text"),
    ("add filter", "look for an entity type"),
    ("remove filter", "stop looking for one"),
    ("preset", "load a preset's filters"),
    ("filters", "show the active filters"),
    ("list filters", "browse every entity type"),
    ("add layer", "switch a detector on"),
    ("remove layer", "switch a detector off"),
    ("layers", "show the active layers"),
    ("serve", "serve this session over HTTP"),
    ("stop-serve", "stop serving"),
    ("clear", "clear the screen"),
    ("help", "show every command"),
    ("quit", "save and leave"),
]


def candidates(text: str, entities: list[str], presets: list[str]) -> list[tuple[str, int, str]]:
    """``(insert, start position, description)`` completions for the prompt so far."""
    if text.startswith("/"):
        fragment = text[1:].lower()
        leading = [(c, d) for c, d in PALETTE if c.startswith(fragment)]
        inside = [
            (c, d)
            for c, d in PALETTE
            if (c, d) not in leading and any(w.startswith(fragment) for w in c.split())
        ]
        return [(cmd, -len(text), desc) for cmd, desc in leading + inside]
    words = text.split()
    ended = text.endswith(" ")
    word = "" if ended else (words[-1] if words else "")
    position = len(words) + (1 if ended else 0)  # 1-based index of the word being typed
    options: list[str] = []
    if position <= 1:
        options = COMMANDS
    elif words[0] in ("add", "remove"):
        if position == 2:
            options = ["filter", "layer"]
        elif position == 3:
            options = ["regex", "ner", "llm"] if words[1] == "layer" else entities
    elif words[0] == "list" and position == 2:
        options = ["filters", "layers"]
    elif words[0] == "preset" and position == 2:
        options = presets
    return [(o, -len(word), "") for o in options if o.startswith(word)]

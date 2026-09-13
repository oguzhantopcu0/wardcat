"""
Load-time ReDoS screening for user-supplied regular expressions.

Python's ``re`` engine holds the GIL for the whole of a match and has no way to
be interrupted, so wrapping a match in a thread with a timeout bounds nothing:
the waiting thread only gets to run again once the match is over. The one place
a runaway pattern can actually be stopped is a separate process, and the one
moment that is cheap enough to do is when the pattern is configured — so this
module screens patterns then, and matching at scan time runs unguarded.

Screening has two steps.

1. The parse tree is searched for the shapes that backtrack exponentially: a
   repeat of variable width nested inside another repeat (``(a+)+``,
   ``(\\w+\\s?)*``), and alternatives inside a repeat that can consume the same
   text (``(a|aa)*``). Most patterns have neither and are accepted with no
   further work.
2. Each suspect is run against inputs built from its shape, in a child
   interpreter the kernel kills after a short deadline. A nested repeat that
   is not actually ambiguous — ``(\\w+\\.)+``, where the dot leaves one way to
   split the text — finishes at once and is accepted.

This is a screen, not a proof. It rejects what it has shown to be catastrophic
and accepts the rest; polynomial backtracking (``\\d+\\d+x``) is not flagged, and
input size is bounded separately by ``max_text_bytes``.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from re import _constants as _c  # type: ignore[attr-defined]
from re import _parser  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

#: How long a suspect pattern may spend on the probe inputs. The slowest-growing
#: exponential shape, ambiguous alternation like ``(a|aa)*``, grows as 1.6**n —
#: some 10**8 steps at this length, far past the deadline — while a quartic one
#: needs about 2.5 million, well inside it.
_DEADLINE_SECONDS = 1.0
_ATTACK_LENGTH = 40
#: Extra wall-clock time allowed for the child interpreter to start. On POSIX the
#: kernel enforces the deadline itself (SIGALRM), so this only matters elsewhere.
_STARTUP_GRACE_SECONDS = 10.0

#: Characters that end a probe run and force the match to fail, so the engine
#: tries every way of splitting the run before giving up.
_SUFFIXES = ("!", "\x00", " ", "a", "1", "\n")

_CATEGORY_SAMPLE = {
    _c.CATEGORY_DIGIT: "1",
    _c.CATEGORY_NOT_DIGIT: "a",
    _c.CATEGORY_SPACE: " ",
    _c.CATEGORY_NOT_SPACE: "a",
    _c.CATEGORY_WORD: "a",
    _c.CATEGORY_NOT_WORD: "!",
}

_REPEATS = (_c.MAX_REPEAT, _c.MIN_REPEAT)

# The child sets an interval timer whose default action terminates the process.
# A signal handler would never run mid-match for the same GIL reason as above;
# the default action needs no Python code to run, so it always does.
_PROBE = """\
import json, re, sys
job = json.loads(sys.stdin.read())
pattern = re.compile(job["pattern"], job["flags"])
try:
    import signal
    signal.setitimer(signal.ITIMER_REAL, job["deadline"])
except (ImportError, AttributeError):
    pass
for text in job["attacks"]:
    pattern.search(text)
"""


def is_catastrophic(pattern: re.Pattern[str]) -> bool:
    """True if *pattern* was shown to backtrack exponentially.

    Patterns with no suspect shape return ``False`` without starting a process.
    When a suspect cannot be probed — no usable interpreter, as in a frozen
    application — it is treated as catastrophic: the caller is configuring a
    guardrail, and refusing a pattern is recoverable where a hung scan is not.
    """
    try:
        tree = _parser.parse(pattern.pattern, pattern.flags)
    except Exception as exc:  # pragma: no cover - private API drift
        logger.warning("Could not screen pattern %r for ReDoS: %s", pattern.pattern, exc)
        return False

    samples = list(dict.fromkeys(_suspects(tree, in_repeat=False)))
    if not samples:
        return False

    prefixes = list(dict.fromkeys(("", _literal_prefix(tree))))
    attacks = [
        prefix + sample * _ATTACK_LENGTH + suffix
        for sample in samples
        for prefix in prefixes
        for suffix in _SUFFIXES
        if suffix != sample
    ]
    return _probe(pattern, attacks)


def _probe(pattern: re.Pattern[str], attacks: list[str]) -> bool:
    if getattr(sys, "frozen", False) or not sys.executable:
        logger.warning(
            "Pattern %r has a nested repeat that could not be probed for ReDoS "
            "(no Python interpreter available) — rejecting it.",
            pattern.pattern,
        )
        return True
    job = json.dumps(
        {
            "pattern": pattern.pattern,
            "flags": pattern.flags,
            "deadline": _DEADLINE_SECONDS,
            "attacks": attacks,
        }
    )
    try:
        done = subprocess.run(
            [sys.executable, "-I", "-S", "-c", _PROBE],
            input=job,
            capture_output=True,
            text=True,
            timeout=_DEADLINE_SECONDS + _STARTUP_GRACE_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return True
    except (OSError, ValueError) as exc:
        logger.warning(
            "Pattern %r could not be probed for ReDoS (%s) — rejecting it.", pattern.pattern, exc
        )
        return True
    if done.returncode == 0:
        return False
    if done.returncode < 0:  # killed by the deadline timer
        return True
    logger.warning(
        "ReDoS probe for pattern %r failed (exit %d): %s — rejecting it.",
        pattern.pattern,
        done.returncode,
        done.stderr.strip()[-200:],
    )
    return True


def _suspects(items: list, *, in_repeat: bool) -> list[str]:
    """Sample characters for every exponential-looking shape in *items*."""
    found: list[str] = []
    for op, av in items:
        if op in _REPEATS:
            low, high, body = av
            if in_repeat and high != low:
                sample = _sample(body)
                if sample is not None:
                    found.append(sample)
            found.extend(_suspects(body, in_repeat=in_repeat or high > 1))
        elif op is _c.POSSESSIVE_REPEAT:
            found.extend(_suspects(av[2], in_repeat=in_repeat))
        elif op is _c.SUBPATTERN:
            found.extend(_suspects(av[-1], in_repeat=in_repeat))
        elif op is _c.ATOMIC_GROUP:
            found.extend(_suspects(av, in_repeat=in_repeat))
        elif op is _c.BRANCH:
            branches = av[1]
            if in_repeat:
                # Two ways to consume the same text: alternatives that start alike,
                # or an empty one — the parser rewrites ``(a|aa)`` as ``a(?:|a)``.
                samples = [_sample(branch) for branch in branches]
                known = [s for s in samples if s is not None]
                if len(known) != len(set(known)) or (known and any(not b for b in branches)):
                    found.append(known[0])
            for branch in branches:
                found.extend(_suspects(branch, in_repeat=in_repeat))
        elif op in (_c.ASSERT, _c.ASSERT_NOT):
            found.extend(_suspects(av[1], in_repeat=in_repeat))
        elif op is _c.GROUPREF_EXISTS:
            _group, yes, no = av
            found.extend(_suspects(yes, in_repeat=in_repeat))
            if no is not None:
                found.extend(_suspects(no, in_repeat=in_repeat))
    return found


def _sample(items: list) -> str | None:
    """A character the start of *items* can match, or ``None`` if unknown."""
    for op, av in items:
        if op is _c.AT:
            continue
        if op is _c.LITERAL:
            return chr(av)
        if op is _c.NOT_LITERAL:
            return "b" if av == ord("a") else "a"
        if op is _c.ANY:
            return "a"
        if op is _c.IN:
            return _sample_set(av)
        if op in _REPEATS or op is _c.POSSESSIVE_REPEAT:
            sample = _sample(av[2])
            if sample is not None or av[0] > 0:
                return sample
            continue  # an optional element: the next one may start the match
        if op is _c.SUBPATTERN:
            return _sample(av[-1])
        if op is _c.ATOMIC_GROUP:
            return _sample(av)
        if op is _c.BRANCH:
            for branch in av[1]:
                sample = _sample(branch)
                if sample is not None:
                    return sample
            return None
        return None
    return None


def _sample_set(items: list) -> str | None:
    if items and items[0][0] is _c.NEGATE:
        excluded = {chr(av) for op, av in items if op is _c.LITERAL}
        excluded |= {chr(av[0]) for op, av in items if op is _c.RANGE}
        return next((ch for ch in "a1 !x" if ch not in excluded), None)
    for op, av in items:
        if op is _c.LITERAL:
            return chr(av)
        if op is _c.RANGE:
            return chr(av[0])
        if op is _c.CATEGORY and av in _CATEGORY_SAMPLE:
            return _CATEGORY_SAMPLE[av]
    return None


def _literal_prefix(items: list) -> str:
    """The literal text a pattern must start with (``"id:"`` for ``^id:\\d+``)."""
    chars: list[str] = []
    for op, av in items:
        if op is _c.AT:
            continue
        if op is not _c.LITERAL:
            break
        chars.append(chr(av))
    return "".join(chars)

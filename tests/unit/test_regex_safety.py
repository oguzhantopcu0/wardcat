"""User-supplied patterns that backtrack catastrophically are refused at configuration.

``re`` cannot be interrupted once a match starts, so a runaway custom or denylist
pattern would hang every scan that reaches it. These tests hold the screen to the
two things it has to do: refuse patterns that really are catastrophic without
hanging while it finds out, and accept the nested-looking patterns people write
every day.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from re import _parser  # type: ignore[attr-defined]

import pytest
import yaml

from wardcat import ConfigError, Entity, Wardcat
from wardcat.config.loader import load_config
from wardcat.utils.regex_safety import _sample, _sample_set, _suspects, is_catastrophic

CATASTROPHIC = [
    r"(a+)+$",
    r"(\w+\s?)+$",  # the old check passed this: it only ever tried a run of "a"
    r"^id:(\d+,?)+$",  # a literal prefix the probe has to supply
    r"(a|aa)*$",  # ambiguous alternation, which the parser rewrites as a(?:|a)
]

# Nested repeats that look suspicious but have one way to split the text.
UNAMBIGUOUS_NESTING = [
    r"(\w+\.)+\w+",
    r"(?:\d{1,3}\.){3}\d{1,3}",
    r"(?:\d{3}[- ]?){2,4}",
]

# No nested repeat at all: accepted without starting a process.
PLAIN = [
    r"\bPROJECT-\d{4}\b",
    r"\b(CEO|CTO|CFO)\b",
    r"[^@\s]+@[^@\s]+\.[a-z]{2,}",
    r"(a++)+$",  # possessive: the inner run can never be split again
]


@pytest.mark.parametrize("pattern", CATASTROPHIC)
def test_catastrophic_patterns_are_caught_quickly(pattern):
    started = time.perf_counter()
    assert is_catastrophic(re.compile(pattern))
    assert time.perf_counter() - started < 15


@pytest.mark.parametrize("pattern", UNAMBIGUOUS_NESTING)
def test_unambiguous_nesting_is_accepted(pattern):
    assert not is_catastrophic(re.compile(pattern))


@pytest.mark.parametrize("pattern", PLAIN)
def test_plain_patterns_never_start_a_process(pattern, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a pattern with no nested repeat was probed")

    monkeypatch.setattr("wardcat.utils.regex_safety.subprocess.run", refuse)
    assert not is_catastrophic(re.compile(pattern))


def test_flags_are_carried_into_the_probe():
    assert is_catastrophic(re.compile(r"(A+)+$", re.IGNORECASE))


def test_a_suspect_that_cannot_be_probed_is_refused(monkeypatch):
    """Refusing a pattern is recoverable; a hung scan is not."""
    monkeypatch.setattr(sys, "executable", "/nonexistent/python")
    assert is_catastrophic(re.compile(UNAMBIGUOUS_NESTING[0]))


def test_frozen_applications_do_not_launch_themselves(monkeypatch):
    """In a frozen app ``sys.executable`` is the app, not an interpreter."""

    def refuse(*args, **kwargs):
        raise AssertionError("probe started in a frozen application")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr("wardcat.utils.regex_safety.subprocess.run", refuse)
    assert is_catastrophic(re.compile(UNAMBIGUOUS_NESTING[0]))


def test_a_probe_that_outlives_its_grace_period_counts_as_catastrophic(monkeypatch):
    """Where the kernel timer is unavailable, the parent's own timeout decides."""

    def hang(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="python", timeout=11)

    monkeypatch.setattr("wardcat.utils.regex_safety.subprocess.run", hang)
    assert is_catastrophic(re.compile(UNAMBIGUOUS_NESTING[0]))


def test_a_probe_that_crashes_is_refused_and_logged(monkeypatch, caplog):
    def crash(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="Traceback: boom"
        )

    monkeypatch.setattr("wardcat.utils.regex_safety.subprocess.run", crash)
    with caplog.at_level(logging.WARNING, logger="wardcat.utils.regex_safety"):
        assert is_catastrophic(re.compile(UNAMBIGUOUS_NESTING[0]))
    assert any("boom" in r.message for r in caplog.records)


# The walker has to see through every construct a nested repeat can hide in, and
# pick a character that construct actually accepts — the probe input is built
# from it, and a character the pattern rejects would make every suspect look safe.
@pytest.mark.parametrize(
    ("pattern", "accepted"),
    [
        (r"(?:[^,]+,?)+$", "a"),  # negated set
        (r"(?:[^a1]+-?)+$", " "),  # negated set excluding the first choices
        (r"(?:[^x]+y?)+$", "a"),  # single negated literal
        (r"(?:[^a]+y?)+$", "b"),
        (r"(?:[c-f]+-?)+$", "c"),  # range
        (r"(?:.+x?)+$", "a"),  # any character
        (r"(?:(?:\bab)+c?)+$", "a"),  # a boundary before the first character
        (r"(?:(a)+b?)+$", "a"),  # capturing group
        (r"(?:(?>ab)+c?)+$", "a"),  # atomic group
        (r"(?:(?:ab|cd)+e?)+$", "a"),  # alternation
        (r"(x)(?:(?:\1?c)+d?)+$", "c"),  # optional element with no known start
        (r"(?=(?:a+b?)+)", "a"),  # lookahead
        (r"(x)?(?(1)(?:a+b?)+|(?:c+d?)+)$", "c"),  # conditional group, "no" branch
        (r"(?:(?>a+b?)+)", "a"),  # atomic group holding the nesting
    ],
)
def test_the_shape_walker_finds_nesting_everywhere(pattern, accepted):
    samples = _suspects(_parser.parse(pattern), in_repeat=False)
    assert accepted in samples
    assert all(re.fullmatch(r".", s, re.DOTALL) for s in samples)


def test_a_start_the_walker_cannot_name_gives_no_sample():
    """A back-reference has no fixed first character, so nothing can be probed."""
    assert _sample([_parser.parse(r"(a)(b)(?:\1|\2)")[-1]]) is None  # alternation
    assert _sample([_parser.parse(r"(a)\1+")[-1]]) is None  # required repeat
    assert _sample_set(_parser.parse(r"[^a1 !x]")[0][1]) is None
    assert _sample_set([]) is None


# ── Where patterns come in ────────────────────────────────────────────────────


def test_yaml_denylist_pattern_is_screened(tmp_path: Path):
    cfg = tmp_path / "policy.yaml"
    cfg.write_text(yaml.dump({"denylist": [{"pattern": r"(a+)+$", "entity_type": "X"}]}))
    with pytest.raises(ConfigError, match="catastrophic backtracking"):
        load_config(cfg)


def test_add_denylist_refuses_a_catastrophic_pattern_and_adds_nothing():
    guard = Wardcat().add_entity(Entity.CUSTOM_SECRET, action="redact")
    guard.add_denylist([{"value": "ProjectX", "entity_type": "CUSTOM_SECRET"}])

    with pytest.raises(ConfigError, match="catastrophic backtracking"):
        guard.add_denylist(
            [
                {"pattern": r"\bPRJ-\d{4}\b", "entity_type": "CUSTOM_SECRET"},
                {"pattern": r"(a+)+$", "entity_type": "CUSTOM_SECRET"},
            ]
        )

    # Validated as a whole: the good entry in the refused call was not added either.
    assert [e.get("value", e.get("pattern")) for e in guard._config["denylist"]] == ["ProjectX"]
    result = guard.scan("ProjectX and PRJ-1234")
    assert result.sanitized_text == "[CUSTOM_SECRET] and PRJ-1234"


def test_add_denylist_refuses_invalid_regex():
    """It used to accept it, and every scan then skipped the entry silently."""
    with pytest.raises(ConfigError, match="not valid regex"):
        Wardcat().add_denylist([{"pattern": "[bad", "entity_type": "CUSTOM_SECRET"}])


def test_denylist_keeps_matching_after_compiling_once():
    guard = (
        Wardcat()
        .add_entity(Entity.CUSTOM_SECRET, action="redact")
        .add_denylist(
            [
                {"pattern": r"(\w+\.)+internal", "entity_type": "CUSTOM_SECRET"},
                {"value": "ProjectX", "entity_type": "CUSTOM_SECRET"},
            ]
        )
    )
    for _ in range(2):
        result = guard.scan("ProjectX runs on build.eu.internal")
        assert result.sanitized_text == "[CUSTOM_SECRET] runs on [CUSTOM_SECRET]"
        assert result.warnings == []

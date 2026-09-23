"""Every example in the README runs, prints what it says it prints, and every link resolves.

The README is also the PyPI long description, so its links must be absolute and
its examples must run offline — no SpaCy model, no LLM backend.
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
DOCS = ROOT / "docs"

_FENCE = re.compile(r"^```(\w+)\n(.*?)^```", re.M | re.S)
_ANY_FENCE = re.compile(r"^```.*?^```\n", re.M | re.S)


def _prose() -> str:
    """The README without its code blocks, so a ``#`` comment is not a heading."""
    return _ANY_FENCE.sub("", README.read_text(encoding="utf-8"))


def _blocks(lang: str) -> list[str]:
    return [body for tag, body in _FENCE.findall(README.read_text(encoding="utf-8")) if tag == lang]


def _expected_output(block: str) -> list[str]:
    """The ``# ...`` comment lines that follow a ``print(`` line, in order."""
    lines = block.splitlines()
    expected: list[str] = []
    for i, line in enumerate(lines):
        if line.startswith("# ") and i > 0 and lines[i - 1].lstrip().startswith("print("):
            expected.append(line[2:])
    return expected


class TestExamplesRun:
    def test_there_are_python_blocks(self) -> None:
        assert 1 <= len(_blocks("python")) <= 2

    @pytest.mark.parametrize("block", _blocks("python"))
    def test_python_block_runs_and_prints_what_it_claims(self, block: str, tmp_path: Path) -> None:
        # The block goes through a file, not the interpreter's stdin: on Windows the
        # venv's python.exe is a trampoline, and a script piped to it never returned.
        script = tmp_path / "readme_block.py"
        script.write_text(block, encoding="utf-8")
        env = dict(os.environ, WARDCAT_SALT="test-salt", PYTHONIOENCODING="utf-8")
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            encoding="utf-8",
            timeout=25,  # under pytest's own 30 s, so a hang is reported as one
            env=env,
            cwd=ROOT,
        )
        assert proc.returncode == 0, proc.stderr
        got = proc.stdout.splitlines()
        for line in _expected_output(block):
            assert line in got, f"README claims {line!r}; stdout was {got!r}"

    def test_cli_teaser(self, capsys, monkeypatch) -> None:
        from wardcat.cli import main

        [block] = [b for b in _blocks("bash") if "wardcat scan" in b]
        command = next(line for line in block.splitlines() if "wardcat scan" in line)
        expected = next(line[2:] for line in block.splitlines() if line.startswith("# "))
        stdin, _, args = command.partition(" | ")
        text = re.match(r'echo "(.*)"', stdin).group(1)
        monkeypatch.setattr(sys, "stdin", io.StringIO(text + "\n"))
        code = main(args.split()[1:])
        assert code == 1  # 1 = something was found
        assert capsys.readouterr().out.strip() == expected


_LINK = re.compile(r"\]\((https?://[^)\s]+|[^)\s]+)\)")
_HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.M)
_NAV_FILES = re.compile(r"^\s+- [^:]+:\s+(\S+\.md)\s*$", re.M)


def _slug(heading: str) -> str:
    text = re.sub(r"`", "", heading).lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", "-", text.strip())


def _docs_anchors(page: Path) -> set[str]:
    prose = _ANY_FENCE.sub("", page.read_text(encoding="utf-8"))
    return {_slug(h) for h in _HEADING.findall(prose)}


class TestLinks:
    def test_no_relative_links(self) -> None:
        relative = [t for t in _LINK.findall(README.read_text(encoding="utf-8")) if "://" not in t]
        assert relative == [], f"PyPI renders the README: links must be absolute, got {relative}"

    @pytest.mark.parametrize(
        "target",
        sorted(
            {
                t
                for t in _LINK.findall(README.read_text(encoding="utf-8"))
                if "docs.wardcat.com" in t
            }
        ),
    )
    def test_docs_link_has_a_page_in_nav_and_its_anchor(self, target: str) -> None:
        path, _, fragment = target.removeprefix("https://docs.wardcat.com").partition("#")
        path = path.strip("/")
        page = DOCS / (f"{path}.md" if path else "index.md")
        assert page.exists(), f"{target} -> {page} is missing"
        nav = _NAV_FILES.findall((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
        assert page.relative_to(DOCS).as_posix() in nav, f"{page.name} is not in the mkdocs nav"
        if fragment:
            assert fragment in _docs_anchors(page), f"{target}: no heading slugs to {fragment!r}"

    @pytest.mark.parametrize(
        "target",
        sorted(
            {
                t
                for t in _LINK.findall(README.read_text(encoding="utf-8"))
                if "github.com/oguzhantopcu0/wardcat/" in t
                and ("/blob/main/" in t or "/tree/main/" in t)
            }
        ),
    )
    def test_repo_link_points_at_a_file_that_exists(self, target: str) -> None:
        rel = re.sub(r"^https://github.com/oguzhantopcu0/wardcat/(blob|tree)/main/", "", target)
        assert (ROOT / rel).exists(), f"{target} -> {rel} is missing in the repository"


class TestBudget:
    """The README defers to the docs site; it does not grow back into one."""

    def test_size(self) -> None:
        text = README.read_text(encoding="utf-8")
        assert len(text.splitlines()) <= 250
        assert len(text.split()) <= 1200

    def test_headings(self) -> None:
        prose = _prose()
        headings = _HEADING.findall(prose)
        levels = [
            len(line) - len(line.lstrip("#"))
            for line in prose.splitlines()
            if re.match(r"^#{1,6}\s", line)
        ]
        assert levels.count(1) == 1, "exactly one H1"
        assert set(levels) <= {1, 2}, "no heading deeper than H2"
        assert len(headings) == len(set(headings)), "no duplicate headings"

    def test_every_fence_has_a_language(self) -> None:
        opening = [
            line
            for line in README.read_text(encoding="utf-8").splitlines()
            if line.startswith("```")
        ][::2]
        assert all(len(line) > 3 for line in opening), opening

    def test_the_disclaimer_survives_verbatim(self) -> None:
        text = README.read_text(encoding="utf-8")
        assert "wardcat is a **best-effort** PII detector" in text
        assert "**not legal advice or a substitute for compliance review**" in text
        assert "Never hardcode the salt in source code or config files." in text
        assert "Inputs exceeding **500 KB** raise a `ValueError`" in text

    def test_no_benchmark_numbers(self) -> None:
        # Results live in benchmarks/, not in the README (they go stale there).
        assert not re.search(
            r"\b(precision|recall|F1)\b.*\d+(\.\d+)?\s*%", README.read_text(encoding="utf-8"), re.I
        )

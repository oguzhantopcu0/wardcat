"""The commands folded in from wardcat-cli: scan's new options, check, is-sensitive.

tests/unit/test_cli.py is the 1.2.x contract and runs unchanged; this file
covers what was added, and the ways the old wardcat-cli failed open.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from wardcat import Wardcat
from wardcat.cli import EXIT_CLEAN, EXIT_CONFIG, EXIT_DEGRADED, EXIT_FOUND, main
from wardcat.cli._guard_args import parse_entity_specs

CARD = "4111 1111 1111 1111"


def run(capsys, *argv, stdin: str | None = None, monkeypatch=None):
    if stdin is not None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    code = main([str(a) for a in argv])
    out, err = capsys.readouterr()
    return code, out, err


def write(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.write_text(text, encoding=encoding)
    return path


class TestEntitySpecs:
    def test_each_item_takes_its_own_action(self) -> None:
        assert parse_entity_specs(["EMAIL,CREDIT_CARD=mask", "IBAN"], "redact") == [
            ("EMAIL", "redact"),
            ("CREDIT_CARD", "mask"),
            ("IBAN", "redact"),
        ]

    @pytest.mark.parametrize("spec", ["EMAIL,", ",EMAIL", "EMAIL=", "=mask"])
    def test_an_empty_item_is_an_error(self, spec, capsys, monkeypatch) -> None:
        code, _, err = run(capsys, "scan", "--entity", spec, stdin="x", monkeypatch=monkeypatch)
        assert code == EXIT_CONFIG and "empty item" in err


class TestScanOptions:
    def test_comma_separated_entities(self, capsys, monkeypatch) -> None:
        code, out, _ = run(
            capsys,
            "scan",
            "--entity",
            "EMAIL,CREDIT_CARD=mask",
            stdin=f"a@b.io {CARD}",
            monkeypatch=monkeypatch,
        )
        assert code == EXIT_FOUND and out == "[EMAIL] ************1111\n"

    def test_group(self, capsys, monkeypatch) -> None:
        code, out, _ = run(
            capsys, "scan", "--group", "financial", stdin=f"card {CARD}", monkeypatch=monkeypatch
        )
        assert code == EXIT_FOUND and "[CREDIT_CARD]" in out

    def test_unknown_group(self, capsys, monkeypatch) -> None:
        code, _, err = run(capsys, "scan", "--group", "nope", stdin="x", monkeypatch=monkeypatch)
        assert code == EXIT_CONFIG and "unknown --group" in err

    def test_output_file(self, tmp_path, capsys, monkeypatch) -> None:
        target = tmp_path / "out.txt"
        code, out, _ = run(
            capsys,
            "scan",
            "--entity",
            "CREDIT_CARD",
            "--output",
            target,
            stdin=f"card {CARD}",
            monkeypatch=monkeypatch,
        )
        assert code == EXIT_FOUND and out == ""
        assert target.read_text(encoding="utf-8") == "card [CREDIT_CARD]\n"

    def test_hash_without_a_salt_warns(self, capsys, monkeypatch) -> None:
        monkeypatch.delenv("WARDCAT_SALT", raising=False)
        code, _, err = run(
            capsys, "scan", "--entity", "CREDIT_CARD=hash", stdin=CARD, monkeypatch=monkeypatch
        )
        assert code == EXIT_FOUND
        assert "no salt" in err and CARD not in err

    def test_the_warning_points_at_an_unused_salt(self, capsys, monkeypatch) -> None:
        monkeypatch.setenv("WARDCAT_SALT", "set-but-unused")
        _, _, err = run(
            capsys, "scan", "--entity", "CREDIT_CARD=hash", stdin=CARD, monkeypatch=monkeypatch
        )
        assert "--salt-env WARDCAT_SALT" in err

    def test_no_warning_with_a_salt(self, capsys, monkeypatch) -> None:
        monkeypatch.setenv("S", "x" * 16)
        _, _, err = run(
            capsys,
            "scan",
            "--entity",
            "CREDIT_CARD=hash",
            "--salt-env",
            "S",
            stdin=CARD,
            monkeypatch=monkeypatch,
        )
        assert "no salt" not in err

    def test_ner_flags_are_exclusive(self, capsys, monkeypatch) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["scan", "--entity", "PERSON", "--ner", "x", "--ner-language", "tr"])
        assert exc.value.code == 2

    def test_adjudicate_needs_an_llm(self, capsys, monkeypatch) -> None:
        code, _, err = run(
            capsys, "scan", "--entity", "EMAIL", "--adjudicate", stdin="x", monkeypatch=monkeypatch
        )
        assert code == EXIT_CONFIG and "--llm" in err

    def test_abbreviated_options_are_refused(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["scan", "--ent", "EMAIL"])
        assert exc.value.code == 2

    def test_version(self, capsys) -> None:
        from wardcat import __version__

        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        assert capsys.readouterr().out.strip() == f"wardcat {__version__}"


class TestCheck:
    def test_finds_and_reports_line_and_column(self, tmp_path, capsys) -> None:
        f = write(tmp_path / "a.txt", f"one\ntwo\nsecret {CARD} here\n")
        code, out, err = run(capsys, "check", f, "--entity", "CREDIT_CARD")
        assert code == EXIT_FOUND
        assert out.startswith(f"{f.as_posix()}:3:8  CREDIT_CARD")
        assert CARD not in out + err

    def test_clean_tree(self, tmp_path, capsys) -> None:
        write(tmp_path / "a.txt", "merhaba dünya\n")
        code, out, _ = run(capsys, "check", tmp_path)
        assert code == EXIT_CLEAN and out == ""

    def test_default_scope_is_everything(self, tmp_path, capsys) -> None:
        write(tmp_path / "k.py", 'KEY = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz"\n')
        code, out, _ = run(capsys, "check", tmp_path)
        assert code == EXIT_FOUND and "CUSTOM_SECRET" in out

    def test_jsonl_and_sarif(self, tmp_path, capsys) -> None:
        write(tmp_path / "c.txt", f"card {CARD}\n")
        code, out, _ = run(capsys, "check", tmp_path, "--format", "jsonl")
        record = json.loads(out.splitlines()[0])
        assert code == EXIT_FOUND and record["entity_type"] == "CREDIT_CARD" and record["line"] == 1
        code, out, _ = run(capsys, "check", tmp_path, "--format", "sarif")
        sarif = json.loads(out)
        assert sarif["runs"][0]["results"][0]["ruleId"] == "CREDIT_CARD"
        assert sarif["runs"][0]["tool"]["driver"]["version"]
        assert CARD not in out

    def test_quiet(self, tmp_path, capsys) -> None:
        write(tmp_path / "c.txt", CARD)
        code, out, err = run(capsys, "check", tmp_path, "--quiet")
        assert code == EXIT_FOUND and out == "" and err == ""

    def test_include_exclude_and_skip_dirs(self, tmp_path, capsys) -> None:
        write(tmp_path / "a.txt", CARD)
        write(tmp_path / "b.log", CARD)
        (tmp_path / "node_modules").mkdir()
        write(tmp_path / "node_modules" / "c.txt", CARD)
        _, out, _ = run(capsys, "check", tmp_path, "--exclude", "*.log")
        assert "a.txt" in out and "b.log" not in out and "node_modules" not in out
        _, out, _ = run(capsys, "check", tmp_path, "--include", "*.log")
        assert "b.log" in out and "a.txt" not in out

    def test_stdin(self, capsys, monkeypatch) -> None:
        code, out, _ = run(
            capsys,
            "check",
            "-",
            "--stdin-filename",
            "patch.py",
            stdin=f"x = '{CARD}'\n",
            monkeypatch=monkeypatch,
        )
        assert code == EXIT_FOUND and out.startswith("patch.py:1:")

    def test_baseline(self, tmp_path, capsys) -> None:
        f = write(tmp_path / "code.py", f"card = '{CARD}'\n")
        baseline = tmp_path / "baseline.json"
        code, _, _ = run(capsys, "check", f, "--baseline", baseline, "--write-baseline")
        assert code == EXIT_CLEAN and baseline.exists()
        assert CARD not in baseline.read_text(encoding="utf-8")
        assert run(capsys, "check", f, "--baseline", baseline)[0] == EXIT_CLEAN
        write(f, f"card = '{CARD}'\nother = 'a@b.io'\n")
        code, out, _ = run(capsys, "check", f, "--baseline", baseline)
        assert code == EXIT_FOUND and "EMAIL" in out and "CREDIT_CARD" not in out

    def test_write_baseline_needs_a_path(self, tmp_path, capsys) -> None:
        assert run(capsys, "check", tmp_path, "--write-baseline")[0] == EXIT_CONFIG

    def test_a_malformed_baseline_is_an_error(self, tmp_path, capsys) -> None:
        bad = write(tmp_path / "b.json", "[not json")
        assert run(capsys, "check", tmp_path, "--baseline", bad)[0] == EXIT_CONFIG


class TestCheckFailsClosed:
    """Every way wardcat-cli 0.5.0 let a file through unscanned."""

    def test_a_missing_path_is_an_error(self, tmp_path, capsys) -> None:
        assert run(capsys, "check", tmp_path / "gone.txt")[0] == EXIT_CONFIG

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
    def test_an_unreadable_file_is_an_error(self, tmp_path, capsys) -> None:
        f = write(tmp_path / "locked.txt", CARD)
        f.chmod(0)
        try:
            code, _, err = run(capsys, "check", tmp_path)
        finally:
            f.chmod(0o600)
        assert code == EXIT_CONFIG and "locked.txt" in err

    def test_a_latin1_file_is_scanned(self, tmp_path, capsys) -> None:
        write(tmp_path / "l1.txt", f"carte bancaire à débiter: {CARD}\n", encoding="latin-1")
        assert run(capsys, "check", tmp_path, "--entity", "CREDIT_CARD")[0] == EXIT_FOUND

    @pytest.mark.parametrize("encoding", ["utf-16", "utf-32", "utf-8-sig"])
    def test_a_file_with_a_bom_is_scanned(self, tmp_path, capsys, encoding) -> None:
        write(tmp_path / "bom.txt", f"card {CARD}\n", encoding=encoding)
        assert run(capsys, "check", tmp_path, "--entity", "CREDIT_CARD")[0] == EXIT_FOUND

    def test_a_binary_file_is_listed_as_skipped(self, tmp_path, capsys) -> None:
        (tmp_path / "blob.bin").write_bytes(b"\x00\x01" + CARD.encode())
        code, _, err = run(capsys, "check", tmp_path)
        assert code == EXIT_CLEAN and "skipped 1 binary file(s)" in err and "blob.bin" in err

    def test_a_large_file_is_split_and_every_piece_scanned(self, tmp_path, capsys) -> None:
        policy = write(
            tmp_path / "p.yaml",
            "max_text_bytes: 4000\nentities:\n  CREDIT_CARD: {enabled: true, action: warn}\n",
        )
        filler = "şğüöçıİ satır\n" * 900  # ~16 KB, four times the limit
        f = write(tmp_path / "big.txt", filler + f"card {CARD}\n" + filler)
        code, out, _ = run(capsys, "check", f, "--config", policy)
        assert code == EXIT_FOUND
        assert f":{filler.count(chr(10)) + 1}:6  CREDIT_CARD" in out

    def test_a_line_longer_than_the_limit_is_an_error(self, tmp_path, capsys) -> None:
        policy = write(
            tmp_path / "p.yaml",
            "max_text_bytes: 4000\nentities:\n  CREDIT_CARD: {enabled: true, action: warn}\n",
        )
        f = write(tmp_path / "one-line.txt", "x" * 10_000 + f" {CARD}")
        code, out, err = run(capsys, "check", f, "--config", policy)
        assert code == EXIT_CONFIG and "longer than the scan limit" in err and out == ""

    def test_a_policy_that_enables_nothing_is_an_error(self, tmp_path, capsys) -> None:
        policy = write(tmp_path / "empty.yaml", "entities: {}\n")
        write(tmp_path / "a.txt", CARD)
        code, _, err = run(capsys, "check", tmp_path, "--config", policy)
        assert code == EXIT_CONFIG and "nothing to scan for" in err

    def test_an_old_wardcat_cli_config_is_refused(self, tmp_path, capsys) -> None:
        old = write(tmp_path / ".wardcat.yaml", "check:\n  entities: CREDIT_CARD\n")
        write(tmp_path / "a.txt", CARD)
        assert run(capsys, "check", tmp_path, "--config", old)[0] == EXIT_CONFIG

    def test_a_planted_config_is_never_picked_up(self, tmp_path, capsys, monkeypatch) -> None:
        write(tmp_path / ".wardcat.yaml", "entities: {}\n")
        monkeypatch.setenv("WARDCAT_CONFIG", str(tmp_path / ".wardcat.yaml"))
        monkeypatch.chdir(tmp_path)
        write(tmp_path / "a.txt", CARD)
        assert run(capsys, "check", "a.txt")[0] == EXIT_FOUND

    def test_strict_degraded_is_three(self, tmp_path, capsys) -> None:
        write(tmp_path / "a.txt", "Ali Veli")
        code = run(
            capsys, "check", tmp_path, "--entity", "PERSON", "--ner", "xx_no_such_model", "--strict"
        )[0]
        assert code == EXIT_DEGRADED


class TestIsSensitive:
    def _patch(self, monkeypatch, *, answer=None, error=None) -> None:
        def fake(self, text):
            if error is not None:
                raise error
            return answer

        monkeypatch.setattr(Wardcat, "is_sensitive", fake)

    def test_sensitive_is_one(self, capsys, monkeypatch) -> None:
        self._patch(monkeypatch, answer=True)
        code, out, _ = run(capsys, "is-sensitive", "--llm", "m", stdin="x", monkeypatch=monkeypatch)
        assert code == EXIT_FOUND and out == "sensitive\n"

    def test_clean_is_zero(self, capsys, monkeypatch) -> None:
        self._patch(monkeypatch, answer=False)
        code, out, _ = run(capsys, "is-sensitive", "--llm", "m", stdin="x", monkeypatch=monkeypatch)
        assert code == EXIT_CLEAN and out == "clean\n"

    @pytest.mark.parametrize("error", [ConnectionError("down"), TimeoutError("slow")])
    def test_an_unreachable_backend_is_three_never_clean(self, capsys, monkeypatch, error) -> None:
        self._patch(monkeypatch, error=error)
        code, out, _ = run(capsys, "is-sensitive", "--llm", "m", stdin="x", monkeypatch=monkeypatch)
        assert code == EXIT_DEGRADED and "clean" not in out

    def test_an_open_circuit_is_three(self, capsys, monkeypatch) -> None:
        from wardcat.llm.circuit import CircuitOpen

        self._patch(monkeypatch, error=CircuitOpen("open", retry_after=30.0))
        code = run(capsys, "is-sensitive", "--llm", "m", stdin="x", monkeypatch=monkeypatch)[0]
        assert code == EXIT_DEGRADED

    def test_a_missing_file_is_two(self, tmp_path, capsys) -> None:
        assert run(capsys, "is-sensitive", tmp_path / "gone", "--llm", "m")[0] == EXIT_CONFIG

    def test_the_model_is_required(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["is-sensitive", "-"])
        assert exc.value.code == 2


class TestPackaging:
    def test_import_wardcat_loads_no_cli_or_server_code(self) -> None:
        code = (
            "import sys, wardcat\n"
            "bad = [m for m in sys.modules if m.split('.')[0] in "
            "('argparse','starlette','uvicorn','fastapi','typer','rich','click','prompt_toolkit')"
            " or m.startswith(('wardcat.cli', 'wardcat.server'))]\n"
            "assert not bad, bad\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_one_console_script(self) -> None:
        from importlib.metadata import entry_points

        scripts = [e for e in entry_points(group="console_scripts") if e.name == "wardcat"]
        assert [e.value for e in scripts] == ["wardcat.cli:main"]

    def test_utf8_round_trip_through_a_real_pipe(self) -> None:
        text = "Şükrü Öztürk, İzmir: kart 4111 1111 1111 1111\n"
        proc = subprocess.run(
            [sys.executable, "-m", "wardcat.cli", "scan", "--entity", "CREDIT_CARD"],
            input=text.encode("utf-8"),
            capture_output=True,
            check=False,
        )
        assert proc.returncode == EXIT_FOUND, proc.stderr
        assert proc.stdout.decode("utf-8").replace("\r\n", "\n") == (
            "Şükrü Öztürk, İzmir: kart [CREDIT_CARD]\n"
        )

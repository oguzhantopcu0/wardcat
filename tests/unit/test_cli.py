"""The wardcat command: exit codes, output that never carries a value."""

from __future__ import annotations

import json

import pytest

from wardcat.cli import EXIT_CLEAN, EXIT_CONFIG, EXIT_DEGRADED, EXIT_FOUND, main

TEXT = "mail ali@example.com, card 4111 1111 1111 1111\n"


def run(capsys, *argv, stdin: str | None = None, monkeypatch=None):
    if stdin is not None:
        import io
        import sys

        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


class TestScan:
    def test_stdin_to_sanitized_text(self, capsys, monkeypatch) -> None:
        code, out, _ = run(
            capsys,
            "scan",
            "-",
            "--entity",
            "EMAIL",
            "--entity",
            "CREDIT_CARD=mask",
            stdin=TEXT,
            monkeypatch=monkeypatch,
        )
        assert code == EXIT_FOUND
        assert out == "mail [EMAIL], card ************1111\n"

    def test_a_file_and_json_output_without_originals(self, capsys, tmp_path) -> None:
        path = tmp_path / "in.txt"
        path.write_text(TEXT)
        code, out, _ = run(capsys, "scan", str(path), "--preset", "pci_dss", "--json")
        assert code == EXIT_FOUND
        data = json.loads(out)
        assert data["is_clean"] is False
        assert "original" not in json.dumps(data)
        assert "4111 1111 1111 1111" not in out

    def test_clean_text_exits_zero(self, capsys, monkeypatch) -> None:
        code, out, _ = run(
            capsys, "scan", "--entity", "EMAIL", stdin="nothing here\n", monkeypatch=monkeypatch
        )
        assert code == EXIT_CLEAN
        assert out == "nothing here\n"

    def test_output_matches_the_library(self, capsys, monkeypatch) -> None:
        from wardcat import Action, Entity, Wardcat

        expected = Wardcat().add_entity(Entity.EMAIL, Action.REDACT).scan(TEXT).sanitized_text
        _, out, _ = run(capsys, "scan", "--entity", "EMAIL", stdin=TEXT, monkeypatch=monkeypatch)
        assert out == expected

    def test_nothing_to_scan_for_is_a_usage_error(self, capsys, monkeypatch) -> None:
        code, _, err = run(capsys, "scan", stdin=TEXT, monkeypatch=monkeypatch)
        assert code == EXIT_CONFIG
        assert "nothing to scan for" in err

    def test_salt_comes_from_the_environment(self, capsys, monkeypatch) -> None:
        monkeypatch.setenv("WC_SALT", "pepper")
        code, out, _ = run(
            capsys,
            "scan",
            "--entity",
            "EMAIL=hash",
            "--salt-env",
            "WC_SALT",
            stdin=TEXT,
            monkeypatch=monkeypatch,
        )
        assert code == EXIT_FOUND
        assert "[EMAIL:" in out

    def test_a_missing_salt_variable_is_refused(self, capsys, monkeypatch) -> None:
        monkeypatch.delenv("WC_SALT", raising=False)
        code, _, err = run(
            capsys,
            "scan",
            "--entity",
            "EMAIL",
            "--salt-env",
            "WC_SALT",
            stdin=TEXT,
            monkeypatch=monkeypatch,
        )
        assert code == EXIT_CONFIG
        assert "WC_SALT" in err

    def test_strict_with_a_missing_model_exits_three(self, capsys, monkeypatch) -> None:
        pytest.importorskip("spacy")
        code, _, err = run(
            capsys,
            "scan",
            "--entity",
            "PERSON",
            "--ner",
            "xx_no_such_model",
            "--strict",
            stdin=TEXT,
            monkeypatch=monkeypatch,
        )
        assert code == EXIT_DEGRADED
        assert "xx_no_such_model" in err


class TestCheckConfig:
    def test_a_valid_file(self, capsys, tmp_path) -> None:
        cfg = tmp_path / "p.yaml"
        cfg.write_text("entities:\n  EMAIL: {enabled: true, action: redact}\n")
        code, out, _ = run(capsys, "check-config", str(cfg))
        assert code == EXIT_CLEAN
        assert "1 entity type(s) enabled" in out

    def test_a_catastrophic_pattern_is_rejected(self, capsys, tmp_path) -> None:
        cfg = tmp_path / "p.yaml"
        cfg.write_text("custom_patterns:\n  BAD: {pattern: '(a+)+$', action: warn}\n")
        code, _, err = run(capsys, "check-config", str(cfg))
        assert code == EXIT_CONFIG
        assert "error:" in err

    def test_a_missing_file(self, capsys, tmp_path) -> None:
        code, _, err = run(capsys, "check-config", str(tmp_path / "nope.yaml"))
        assert code == EXIT_CONFIG


class TestEntities:
    def test_lists_a_layer(self, capsys) -> None:
        code, out, _ = run(capsys, "entities", "--layer", "ner")
        assert code == EXIT_CLEAN
        assert "PERSON" in out.split()
        assert "EMAIL" not in out.split()

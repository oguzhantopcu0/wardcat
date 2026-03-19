"""CLI testleri — subprocess yerine direkt fonksiyon çağrısı."""
import json
from pathlib import Path

import pytest

from ai_guard.__main__ import _build_parser, cmd_scan, cmd_batch


def parse(args: list[str]):
    return _build_parser().parse_args(args)


class TestParser:
    def test_scan_text(self):
        args = parse(["scan", "--text", "merhaba"])
        assert args.command == "scan"
        assert args.text == "merhaba"

    def test_scan_no_ner_flag(self):
        args = parse(["scan", "--text", "x", "--no-ner"])
        assert args.no_ner is True

    def test_scan_json_format(self):
        args = parse(["scan", "--text", "x", "--format", "json"])
        assert args.format == "json"

    def test_batch_requires_file(self):
        with pytest.raises(SystemExit):
            parse(["batch"])


class TestCmdScan:
    def test_text_output(self, capsys):
        args = parse(["scan", "--text", "merhaba", "--no-ner"])
        cmd_scan(args)
        out = capsys.readouterr().out
        assert "merhaba" in out

    def test_json_output(self, capsys):
        args = parse(["scan", "--text", "kart: 4111111111111111", "--no-ner", "--format", "json"])
        cmd_scan(args)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "violations" in data
        assert any(v["entity_type"] == "CREDIT_CARD" for v in data["violations"])

    def test_scan_from_file(self, tmp_path: Path, capsys):
        f = tmp_path / "input.txt"
        f.write_text("email: test@example.com")
        args = parse(["scan", "--file", str(f), "--no-ner"])
        cmd_scan(args)
        out = capsys.readouterr().out
        assert "EMAIL" in out

    def test_missing_file_raises(self):
        args = parse(["scan", "--file", "/tmp/yok_12345.txt", "--no-ner"])
        with pytest.raises(FileNotFoundError):
            cmd_scan(args)


class TestCmdBatch:
    def test_batch_json(self, tmp_path: Path, capsys):
        f = tmp_path / "lines.txt"
        f.write_text("a@b.com\ntemiz metin\n4111111111111111")
        args = parse(["batch", "--file", str(f), "--no-ner", "--format", "json"])
        cmd_batch(args)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert len(data) == 3
        assert data[0]["violations"][0]["entity_type"] == "EMAIL"
        assert data[1]["is_clean"] is True

    def test_batch_text_output(self, tmp_path: Path, capsys):
        f = tmp_path / "lines.txt"
        f.write_text("a@b.com\ntemiz")
        args = parse(["batch", "--file", str(f), "--no-ner"])
        cmd_batch(args)
        out = capsys.readouterr().out
        assert "Satır 1" in out
        assert "Satır 2" in out

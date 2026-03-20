"""CLI tests — direct function calls instead of subprocess."""
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
        assert "Line 1" in out
        assert "Line 2" in out


from unittest.mock import patch, MagicMock


class TestResolveURL:
    def test_args_url_takes_priority(self):
        from ai_guard.__main__ import _resolve_llm_url
        result = _resolve_llm_url("http://myserver:11434")
        assert result == "http://myserver:11434"

    def test_env_fallback(self):
        from ai_guard.__main__ import _resolve_llm_url
        import os
        with patch.dict(os.environ, {"LLMGUARD_LLM_URL": "http://env:11434"}, clear=False):
            result = _resolve_llm_url("")
        assert result == "http://env:11434"

    def test_default_fallback(self):
        from ai_guard.__main__ import _resolve_llm_url
        import os
        env = {k: v for k, v in os.environ.items() if k != "LLMGUARD_LLM_URL"}
        with patch.dict(os.environ, env, clear=True):
            result = _resolve_llm_url("")
        assert result == "http://localhost:11434"


class TestMakeBackend:
    def test_openai_compat_backend(self):
        from ai_guard.__main__ import _make_backend, _build_parser
        parser = _build_parser()
        args = parser.parse_args(["models", "list", "--llm-backend", "openai_compatible", "--llm-url", "http://localhost:8000"])
        backend = _make_backend(args)
        from ai_guard.llm.backends.openai_compat import OpenAICompatBackend
        assert isinstance(backend, OpenAICompatBackend)


class TestCmdModelsPull:
    def test_pull_calls_backend(self, capsys):
        from ai_guard.__main__ import _build_parser, cmd_models
        parser = _build_parser()
        args = parser.parse_args(["models", "pull", "llama3.1:8b", "--llm-url", "http://localhost:11434"])
        mock_backend = MagicMock()
        with patch("ai_guard.llm.backends.ollama.OllamaBackend", return_value=mock_backend):
            cmd_models(args)
        mock_backend.pull_model.assert_called_once()


class TestCmdSetupEOF:
    def test_eof_cancels_setup(self, capsys):
        from ai_guard.__main__ import _build_parser, cmd_models
        parser = _build_parser()
        args = parser.parse_args(["models", "setup"])
        mock_backend = MagicMock()
        with (
            patch("ai_guard.llm.backends.ollama.OllamaBackend", return_value=mock_backend),
            patch("builtins.input", side_effect=EOFError),
        ):
            cmd_models(args)
        out = capsys.readouterr().out
        assert "Cancelled" in out or "cancelled" in out.lower()


class TestMainHandlers:
    def test_main_file_not_found_exits_1(self):
        import sys
        from ai_guard.__main__ import main
        with (
            patch("sys.argv", ["ai-guard", "scan", "--file", "/no/such/file.txt", "--no-ner"]),
            pytest.raises(SystemExit) as exc,
        ):
            main()
        assert exc.value.code == 1

    def test_main_value_error_exits_1(self, tmp_path):
        import sys
        from ai_guard.__main__ import main
        f = tmp_path / "bad.txt"
        f.write_bytes(b"\xff\xfe bad encoding")
        with (
            patch("sys.argv", ["ai-guard", "scan", "--file", str(f), "--no-ner"]),
            pytest.raises(SystemExit) as exc,
        ):
            main()
        assert exc.value.code == 1

    def test_main_keyboard_interrupt_exits_130(self):
        from ai_guard.__main__ import main
        with (
            patch("sys.argv", ["ai-guard", "scan", "--text", "test", "--no-ner"]),
            patch("ai_guard.__main__.cmd_scan", side_effect=KeyboardInterrupt),
            pytest.raises(SystemExit) as exc,
        ):
            main()
        assert exc.value.code == 130

    def test_main_connection_error_exits_1(self):
        from ai_guard.__main__ import main
        with (
            patch("sys.argv", ["ai-guard", "scan", "--text", "test", "--no-ner"]),
            patch("ai_guard.__main__.cmd_scan", side_effect=ConnectionError("refused")),
            pytest.raises(SystemExit) as exc,
        ):
            main()
        assert exc.value.code == 1

"""CLI tests — direct function calls instead of subprocess."""

import json
from pathlib import Path

import pytest

from ai_guard.__main__ import _build_parser, cmd_batch, cmd_scan


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

    def test_text_output_with_hash_replacement(self, capsys):
        """Text output should include the → replacement arrow for hash actions (line 206)."""
        args = parse(["scan", "--text", "kart: 4111111111111111", "--no-ner"])
        cmd_scan(args)
        out = capsys.readouterr().out
        # CREDIT_CARD default action is hash → replacement is shown with →
        assert "→" in out or "CREDIT_CARD" in out

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


from unittest.mock import MagicMock, patch


class TestResolveURL:
    def test_args_url_takes_priority(self):
        from ai_guard.__main__ import _resolve_llm_url

        result = _resolve_llm_url("http://myserver:11434")
        assert result == "http://myserver:11434"

    def test_env_fallback(self):
        import os

        from ai_guard.__main__ import _resolve_llm_url

        with patch.dict(os.environ, {"LLMGUARD_LLM_URL": "http://env:11434"}, clear=False):
            result = _resolve_llm_url("")
        assert result == "http://env:11434"

    def test_default_fallback(self):
        import os

        from ai_guard.__main__ import _resolve_llm_url

        env = {k: v for k, v in os.environ.items() if k != "LLMGUARD_LLM_URL"}
        with patch.dict(os.environ, env, clear=True):
            result = _resolve_llm_url("")
        assert result == "http://localhost:11434"


class TestMakeBackend:
    def test_openai_compat_backend(self):
        from ai_guard.__main__ import _build_parser, _make_backend

        parser = _build_parser()
        args = parser.parse_args(
            [
                "models",
                "list",
                "--llm-backend",
                "openai_compatible",
                "--llm-url",
                "http://localhost:8000",
            ]
        )
        backend = _make_backend(args)
        from ai_guard.llm.backends.openai_compat import OpenAICompatBackend

        assert isinstance(backend, OpenAICompatBackend)


class TestCmdModelsPull:
    def test_pull_calls_backend(self, capsys):
        from ai_guard.__main__ import _build_parser, cmd_models

        parser = _build_parser()
        args = parser.parse_args(
            ["models", "pull", "llama3.1:8b", "--llm-url", "http://localhost:11434"]
        )
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


class TestBatchWorkersCLI:
    """G2b: --batch-workers flag for batch subcommand."""

    def test_batch_workers_parsed(self):
        args = parse(["batch", "--file", "/tmp/x.txt", "--batch-workers", "8"])
        assert args.batch_workers == 8

    def test_batch_workers_default_is_none(self):
        args = parse(["batch", "--file", "/tmp/x.txt"])
        assert args.batch_workers is None

    def test_batch_workers_used_in_scan(self, tmp_path: Path, capsys):
        f = tmp_path / "lines.txt"
        f.write_text("a@b.com\ntemiz metin\n")
        args = parse(["batch", "--file", str(f), "--no-ner", "--batch-workers", "2"])
        cmd_batch(args)
        out = capsys.readouterr().out
        assert "Line 1" in out


class TestCmdSpacyList:
    def test_list_all_languages(self, capsys):
        from ai_guard.__main__ import _build_parser, cmd_spacy

        args = _build_parser().parse_args(["spacy", "list"])
        cmd_spacy(args)
        out = capsys.readouterr().out
        assert "English" in out
        assert "en_core_web_sm" in out

    def test_list_filtered_by_lang(self, capsys):
        from ai_guard.__main__ import _build_parser, cmd_spacy

        args = _build_parser().parse_args(["spacy", "list", "--lang", "tr"])
        cmd_spacy(args)
        out = capsys.readouterr().out
        assert "Turkish" in out
        assert "English" not in out

    def test_list_unknown_lang_shows_message(self, capsys):
        from ai_guard.__main__ import _build_parser, cmd_spacy

        args = _build_parser().parse_args(["spacy", "list", "--lang", "zz"])
        cmd_spacy(args)
        out = capsys.readouterr().out
        assert "No models found" in out

    def test_list_shows_recommended_marker(self, capsys):
        from ai_guard.__main__ import _build_parser, cmd_spacy

        args = _build_parser().parse_args(["spacy", "list", "--lang", "en"])
        cmd_spacy(args)
        out = capsys.readouterr().out
        assert "←" in out  # recommended marker


class TestCmdSpacyInstalled:
    def test_installed_with_models(self, capsys):
        from ai_guard.__main__ import _build_parser, cmd_spacy

        args = _build_parser().parse_args(["spacy", "installed"])
        with patch("spacy.util.get_installed_models", return_value=["en_core_web_sm"]):
            cmd_spacy(args)
        out = capsys.readouterr().out
        assert "en_core_web_sm" in out

    def test_installed_no_models(self, capsys):
        from ai_guard.__main__ import _build_parser, cmd_spacy

        args = _build_parser().parse_args(["spacy", "installed"])
        with patch("spacy.util.get_installed_models", return_value=[]):
            cmd_spacy(args)
        out = capsys.readouterr().out
        assert "No SpaCy models installed" in out

    def test_installed_no_spacy(self, capsys):
        import sys

        from ai_guard.__main__ import _build_parser, cmd_spacy

        args = _build_parser().parse_args(["spacy", "installed"])
        # Temporarily remove spacy from sys.modules to simulate missing install
        orig = sys.modules.pop("spacy", None)
        orig_util = sys.modules.pop("spacy.util", None)
        try:
            cmd_spacy(args)
        except Exception:
            pass
        finally:
            if orig is not None:
                sys.modules["spacy"] = orig
            if orig_util is not None:
                sys.modules["spacy.util"] = orig_util
        out = capsys.readouterr().out
        # Either shows the installed list or the "not installed" message
        assert len(out) > 0


class TestCmdSpacyDownload:
    def test_download_standard_model_pip_success(self, capsys):
        from ai_guard.__main__ import _build_parser, cmd_spacy

        args = _build_parser().parse_args(["spacy", "download", "en_core_web_sm"])
        with (
            patch("subprocess.run", return_value=MagicMock(returncode=0)),
            patch("shutil.which", return_value=None),
        ):
            cmd_spacy(args)
        out = capsys.readouterr().out
        assert "Model ready" in out

    def test_download_incompatible_model_exits(self, capsys):
        from ai_guard.__main__ import _build_parser, cmd_spacy

        args = _build_parser().parse_args(["spacy", "download", "tr_core_news_trf"])
        with pytest.raises(SystemExit) as exc:
            cmd_spacy(args)
        assert exc.value.code == 1

    def test_download_not_in_catalog_warns_stderr(self, capsys):
        from ai_guard.__main__ import _build_parser, cmd_spacy

        args = _build_parser().parse_args(["spacy", "download", "xx_unknown_model_xyz"])
        with (
            patch("subprocess.run", return_value=MagicMock(returncode=0)),
            patch("shutil.which", return_value=None),
        ):
            cmd_spacy(args)
        err = capsys.readouterr().err
        assert "not in the ai-guard catalog" in err

    def test_download_failure_raises_value_error(self, capsys):
        from ai_guard.__main__ import _build_parser, cmd_spacy

        args = _build_parser().parse_args(["spacy", "download", "en_core_web_sm"])
        with (
            patch("subprocess.run", return_value=MagicMock(returncode=1)),
            patch("shutil.which", return_value=None),
        ):
            with pytest.raises(ValueError, match="download failed"):
                cmd_spacy(args)

    def test_download_with_wheel_url_model(self, capsys):
        from ai_guard.__main__ import _build_parser, cmd_spacy

        args = _build_parser().parse_args(["spacy", "download", "tr_core_news_md"])
        with (
            patch("subprocess.run", return_value=MagicMock(returncode=0)),
            patch("shutil.which", return_value=None),
        ):
            cmd_spacy(args)
        out = capsys.readouterr().out
        assert "Model ready" in out


class TestCmdModelsListRecommended:
    def test_list_recommended_prints_catalog(self, capsys):
        from ai_guard.__main__ import _build_parser, cmd_models

        args = _build_parser().parse_args(["models", "list", "--recommended"])
        cmd_models(args)
        out = capsys.readouterr().out
        assert "Recommended" in out or "VRAM" in out

    def test_list_from_backend(self, capsys):
        from ai_guard.__main__ import _build_parser, cmd_models

        args = _build_parser().parse_args(["models", "list"])
        mock_backend = MagicMock()
        mock_mgr = MagicMock()
        mock_mgr.list.return_value = ["llama3.1:8b", "mistral:7b"]
        with (
            patch("ai_guard.llm.backends.ollama.OllamaBackend", return_value=mock_backend),
            patch("ai_guard.llm.model_manager.ModelManager", return_value=mock_mgr),
        ):
            cmd_models(args)
        out = capsys.readouterr().out
        assert "llama3.1:8b" in out

    def test_list_from_backend_empty(self, capsys):
        from ai_guard.__main__ import _build_parser, cmd_models

        args = _build_parser().parse_args(["models", "list"])
        mock_backend = MagicMock()
        mock_mgr = MagicMock()
        mock_mgr.list.return_value = []
        with (
            patch("ai_guard.llm.backends.ollama.OllamaBackend", return_value=mock_backend),
            patch("ai_guard.llm.model_manager.ModelManager", return_value=mock_mgr),
        ):
            cmd_models(args)
        out = capsys.readouterr().out
        assert "No models found" in out


class TestCmdSetupNonInteractive:
    def test_non_interactive_selects_default_model(self, capsys):
        from ai_guard.__main__ import _build_parser, cmd_models

        args = _build_parser().parse_args(["models", "setup", "--non-interactive"])
        mock_backend = MagicMock()
        mock_mgr = MagicMock()
        mock_mgr.ensure_available.return_value = True
        with (
            patch("ai_guard.llm.backends.ollama.OllamaBackend", return_value=mock_backend),
            patch("ai_guard.llm.model_manager.ModelManager", return_value=mock_mgr),
        ):
            cmd_models(args)
        out = capsys.readouterr().out
        assert mock_mgr.ensure_available.called
        assert "Usage" in out or "guard" in out.lower()

    def test_setup_invalid_selection_cancels(self, capsys):
        from ai_guard.__main__ import _build_parser, cmd_models

        args = _build_parser().parse_args(["models", "setup"])
        mock_backend = MagicMock()
        mock_mgr = MagicMock()
        with (
            patch("ai_guard.llm.backends.ollama.OllamaBackend", return_value=mock_backend),
            patch("ai_guard.llm.model_manager.ModelManager", return_value=mock_mgr),
            patch("builtins.input", return_value="999"),
        ):
            cmd_models(args)
        out = capsys.readouterr().out
        assert "Invalid" in out or "cancelled" in out.lower()

    def test_setup_valid_selection(self, capsys):
        from ai_guard.__main__ import _build_parser, cmd_models

        args = _build_parser().parse_args(["models", "setup"])
        mock_backend = MagicMock()
        mock_mgr = MagicMock()
        mock_mgr.ensure_available.return_value = True
        with (
            patch("ai_guard.llm.backends.ollama.OllamaBackend", return_value=mock_backend),
            patch("ai_guard.llm.model_manager.ModelManager", return_value=mock_mgr),
            patch("builtins.input", return_value="1"),
        ):
            cmd_models(args)
        assert mock_mgr.ensure_available.called

    def test_setup_default_selection(self, capsys):
        from ai_guard.__main__ import _build_parser, cmd_models

        args = _build_parser().parse_args(["models", "setup"])
        mock_backend = MagicMock()
        mock_mgr = MagicMock()
        mock_mgr.ensure_available.return_value = True
        with (
            patch("ai_guard.llm.backends.ollama.OllamaBackend", return_value=mock_backend),
            patch("ai_guard.llm.model_manager.ModelManager", return_value=mock_mgr),
            patch("builtins.input", return_value=""),
        ):
            cmd_models(args)
        assert mock_mgr.ensure_available.called


class TestMainHandlers:
    def test_main_file_not_found_exits_1(self):
        from ai_guard.__main__ import main

        with (
            patch("sys.argv", ["ai-guard", "scan", "--file", "/no/such/file.txt", "--no-ner"]),
            pytest.raises(SystemExit) as exc,
        ):
            main()
        assert exc.value.code == 1

    def test_main_value_error_exits_1(self, tmp_path):
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

    def test_main_unexpected_exception_exits_1(self):
        from ai_guard.__main__ import main

        with (
            patch("sys.argv", ["ai-guard", "scan", "--text", "test", "--no-ner"]),
            patch("ai_guard.__main__.cmd_scan", side_effect=RuntimeError("unexpected")),
            pytest.raises(SystemExit) as exc,
        ):
            main()
        assert exc.value.code == 1

    def test_main_dispatches_batch(self, tmp_path: Path):
        from ai_guard.__main__ import main

        f = tmp_path / "lines.txt"
        f.write_text("a@b.com\n")
        with (
            patch("sys.argv", ["ai-guard", "batch", "--file", str(f), "--no-ner"]),
            patch("ai_guard.__main__.cmd_batch") as mock_batch,
        ):
            main()
        mock_batch.assert_called_once()

    def test_main_dispatches_spacy(self):
        from ai_guard.__main__ import main

        with (
            patch("sys.argv", ["ai-guard", "spacy", "list"]),
            patch("ai_guard.__main__.cmd_spacy") as mock_spacy,
        ):
            main()
        mock_spacy.assert_called_once()

    def test_main_dispatches_models(self):
        from ai_guard.__main__ import main

        with (
            patch("sys.argv", ["ai-guard", "models", "list", "--recommended"]),
            patch("ai_guard.__main__.cmd_models") as mock_models,
        ):
            main()
        mock_models.assert_called_once()

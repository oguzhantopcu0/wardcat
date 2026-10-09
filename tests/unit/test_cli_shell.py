"""The interactive screen: what it does, and what it keeps out of files."""

from __future__ import annotations

import argparse
import io
import json
import os
import stat
import sys

import pytest

from wardcat.cli import EXIT_CLEAN, EXIT_CONFIG, main
from wardcat.cli import _sessions as sessions
from wardcat.cli._shell_ui import candidates
from wardcat.cli.shell import Shell, history_skips

CARD = "4111 1111 1111 1111"


@pytest.fixture(autouse=True)
def state_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "state"))
    return tmp_path / "state"


def answers(monkeypatch, *replies: str) -> None:
    """Feed the guided LLM setup: backend, model, host, port, API key."""
    it = iter(replies)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))


GUIDED = ("openai_compatible", "m", "localhost", "8000", "typed-api-key-5678")


def screen(capsys, *lines: str, resume: str | None = None, cont: bool = False, salt_env=None):
    args = argparse.Namespace(screen_resume=resume, screen_continue=cont, screen_salt_env=salt_env)
    shell = Shell.start(args, interactive=False)
    feed = iter(lines)

    def read() -> str:
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None

    shell.loop(read)
    return shell, capsys.readouterr().out


class TestScan:
    def test_default_filters_find_and_show_no_value(self, capsys) -> None:
        _, out = screen(capsys, f"scan card {CARD} mail ali@example.com")
        assert "card [CREDIT_CARD] mail [EMAIL]" in out
        assert CARD not in out and "ali@example.com" not in out
        assert "CREDIT_CARD" in out and "regex" in out

    def test_nothing_found(self, capsys) -> None:
        assert "nothing found" in screen(capsys, "scan hello")[1]

    def test_an_unscannable_text_shows_nothing(self, capsys, monkeypatch) -> None:
        from wardcat import Wardcat

        class Failed:
            scan_error = "too large"
            sanitized_text = "raw " + CARD
            violations: list = []
            warnings: list = []

        monkeypatch.setattr(Wardcat, "scan", lambda self, text: Failed())
        out = screen(capsys, "scan x")[1]
        assert "could not be scanned" in out and CARD not in out

    def test_a_backend_failure_keeps_the_screen_and_hides_the_message(
        self, capsys, monkeypatch
    ) -> None:
        from wardcat import Wardcat

        def down(self, text):
            raise RuntimeError(f"backend said: {text}")

        monkeypatch.setattr(Wardcat, "scan", down)
        _, out = screen(capsys, f"scan {CARD}", "help")
        assert "error: RuntimeError" in out and CARD not in out and "preset NAME" in out


class TestFiltersAndLayers:
    def test_add_with_actions_then_remove(self, capsys) -> None:
        shell, out = screen(
            capsys, "add filter iban,email=mask --action hash", "remove filter IBAN"
        )
        assert shell.state["filters"]["EMAIL"] == "mask"
        assert "IBAN" not in shell.state["filters"]
        assert "no salt" in out

    def test_an_unknown_type_suggests_and_changes_nothing(self, capsys) -> None:
        shell, out = screen(capsys, "add filter EMIAL")
        assert "did you mean EMAIL" in out and "EMIAL" not in shell.state["filters"]

    def test_a_bad_action_changes_nothing(self, capsys) -> None:
        shell, out = screen(capsys, "add filter EMAIL --action shred")
        assert "error" in out and shell.state["filters"]["EMAIL"] == "redact"

    def test_preset_replaces_the_filters(self, capsys) -> None:
        shell, out = screen(capsys, "preset kvkk")
        assert shell.state["filters"]["TC_ID"] == "hash" and "JWT" not in shell.state["filters"]
        assert "only found with the ner and llm layers" in out

    def test_one_layer_must_stay(self, capsys) -> None:
        shell, out = screen(capsys, "remove layer regex")
        assert "one layer must stay on" in out and shell.state["layers"] == ["regex"]

    def test_ner_needs_a_language_the_first_time(self, capsys) -> None:
        shell, out = screen(capsys, "add layer ner")
        assert "add layer ner tr" in out and shell.state["ner"] is None

    def test_llm_by_flags_reads_the_key_from_the_environment(self, capsys, monkeypatch) -> None:
        monkeypatch.setenv("LLM_KEY", "sk-test-value")
        shell, _ = screen(
            capsys,
            "add layer llm --model m --backend openai_compatible "
            "--base-url https://llm.example/v1 --api-key-env LLM_KEY",
        )
        assert shell.state["layers"] == ["regex", "llm"]
        assert shell.state["llm"]["api_key_env"] == "LLM_KEY"

    def test_unknown_command_and_help_stay_in_the_screen(self, capsys) -> None:
        _, out = screen(capsys, "scna", "add layer ner -h", "help")
        assert "did you mean scan" in out and "--language" in out and "preset NAME" in out


class TestNothingSecretOnDisk:
    def _file(self, state_home, shell) -> str:
        path = state_home / "wardcat" / "sessions" / f"{shell.state['id']}.json"
        return path.read_text(encoding="utf-8")

    def test_the_session_file_has_no_text_salt_or_key(
        self, capsys, monkeypatch, state_home
    ) -> None:
        monkeypatch.setenv("MY_SALT", "pepper-salt-value-1234")
        answers(monkeypatch, *GUIDED)
        shell, _ = screen(capsys, f"scan card {CARD}", "add layer llm", salt_env="MY_SALT")
        assert shell.api_key == "typed-api-key-5678"
        assert shell.state["llm"]["base_url"] == "http://localhost:8000/v1"
        saved = self._file(state_home, shell)
        for secret in (CARD, "pepper-salt-value-1234", "typed-api-key-5678"):
            assert secret not in saved
        assert json.loads(saved)["salt_env"] == "MY_SALT"
        assert json.loads(saved)["llm"]["key_typed"] is True

    @pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
    def test_files_and_folders_are_owner_only(self, capsys, state_home) -> None:
        shell, _ = screen(capsys, "help")
        folder = state_home / "wardcat" / "sessions"
        assert stat.S_IMODE(folder.stat().st_mode) == 0o700
        assert stat.S_IMODE((folder / f"{shell.state['id']}.json").stat().st_mode) == 0o600

    @pytest.mark.parametrize(
        ("line", "skipped"),
        [("scan 4111", True), ("/scan x", True), ("  SCAN x", True), ("add filter EMAIL", False)],
    )
    def test_history_never_keeps_a_scan_line(self, line, skipped) -> None:
        assert history_skips(line) is skipped


class TestSessions:
    def test_resume_keeps_the_setup(self, capsys) -> None:
        first, _ = screen(capsys, "add filter IBAN=mask")
        again, out = screen(capsys, resume=first.state["id"])
        assert again.state["filters"]["IBAN"] == "mask" and "Resuming" in out

    def test_continue_takes_the_latest(self, capsys) -> None:
        first, _ = screen(capsys, "help")
        again, _ = screen(capsys, cont=True)
        assert again.state["id"] == first.state["id"]

    def test_a_typed_key_is_not_there_on_resume(self, capsys, monkeypatch) -> None:
        answers(monkeypatch, *GUIDED)
        first, _ = screen(capsys, "add layer llm")
        again, out = screen(capsys, resume=first.state["id"])
        assert "llm" not in again.state["layers"] and "not saved" in out

    def test_a_wardcat_cli_file_loses_its_salt(self, capsys, state_home) -> None:
        folder = state_home / "wardcat" / "sessions"
        folder.mkdir(parents=True)
        old = {
            "schema": 1,
            "id": "abcd1234",
            "salt": "old-plaintext-salt",
            "filters": {"EMAIL": "hash"},
            "active_layers": ["regex"],
        }
        (folder / "abcd1234.json").write_text(json.dumps(old), encoding="utf-8")
        shell, out = screen(capsys, resume="abcd1234")
        assert "salt is not reused" in out and shell.salt == ""
        assert "old-plaintext-salt" not in (folder / "abcd1234.json").read_text(encoding="utf-8")

    @pytest.mark.parametrize("bad", ["../x", "a/b", "", "x" * 65])
    def test_ids_cannot_leave_the_folder(self, bad) -> None:
        from wardcat.exceptions import ConfigError

        with pytest.raises(ConfigError):
            sessions.path_for(bad)

    def test_sessions_command(self, capsys) -> None:
        first, _ = screen(capsys, "help")
        assert main(["sessions"]) == EXIT_CLEAN
        assert first.state["id"] in capsys.readouterr().out
        assert main(["sessions", "--delete", first.state["id"]]) == EXIT_CLEAN
        assert main(["sessions", "--delete", first.state["id"]]) == EXIT_CONFIG


class TestOpening:
    def test_outside_a_terminal_bare_wardcat_is_a_usage_error(self, capsys, monkeypatch) -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        assert main([]) == EXIT_CONFIG
        assert "needs a terminal" in capsys.readouterr().err

    def test_in_a_terminal_bare_wardcat_opens_the_screen(self, monkeypatch) -> None:
        import wardcat.cli.shell as shell_module

        opened = {}
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)

        class Started:
            def loop(self) -> None:
                opened["looped"] = True

        monkeypatch.setattr(
            shell_module.Shell, "start", classmethod(lambda cls, a, interactive: Started())
        )
        assert main([]) == EXIT_CLEAN and opened == {"looped": True}

    def test_screen_options_with_a_command_are_refused(self, capsys) -> None:
        assert main(["--continue", "entities"]) == EXIT_CONFIG


class TestGuidedLlm:
    def test_a_remote_host_over_http_needs_a_yes(self, capsys, monkeypatch) -> None:
        answers(monkeypatch, "ollama", "m", "10.0.0.5", "11434", "n")
        shell, out = screen(capsys, "add layer llm")
        assert "unencrypted" in out and "cancelled" in out and shell.state["llm"] is None

    def test_a_remote_host_with_a_yes(self, capsys, monkeypatch) -> None:
        answers(monkeypatch, "ollama", "m", "10.0.0.5", "11434", "y", "")
        shell, _ = screen(capsys, "add layer llm")
        assert shell.state["llm"]["base_url"] == "http://10.0.0.5:11434"
        assert shell.state["llm"]["allow_http"] is True and shell.api_key == ""


class TestPalette:
    def test_slash_lists_commands_with_prefix_matches_first(self) -> None:
        items = candidates("/fil", [], [])
        assert items[0][0] == "filters" and ("add filter", -4, "look for an entity type") in items

    def test_words_complete_by_position(self) -> None:
        entities = ["EMAIL", "IBAN"]
        assert [c[0] for c in candidates("add filter E", entities, [])] == ["EMAIL"]
        assert [c[0] for c in candidates("add la", entities, [])] == ["layer"]
        assert [c[0] for c in candidates("add layer ", entities, [])] == ["regex", "ner", "llm"]
        assert [c[0] for c in candidates("preset k", entities, ["gdpr", "kvkk"])] == ["kvkk"]


class TestServeFromTheScreen:
    def test_serves_a_snapshot_and_restarts_on_a_change(self, capsys) -> None:
        pytest.importorskip("uvicorn")
        import socket

        import httpx

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        args = argparse.Namespace(screen_resume=None, screen_continue=False, screen_salt_env=None)
        shell = Shell.start(args, interactive=False)
        try:
            shell.dispatch("serve", f"--port {port}")
            url = f"http://127.0.0.1:{port}/scan"
            body = httpx.post(url, json={"text": "iban TR330006100519786457841326"}).json()
            assert body["sanitized_text"] == "iban [IBAN]"
            shell.dispatch("remove", "filter IBAN")  # restarts with the new policy
            body = httpx.post(url, json={"text": "iban TR330006100519786457841326"}).json()
            assert body["is_clean"] is True
            assert "restarted" in capsys.readouterr().out
        finally:
            shell.dispatch("stop-serve", "")
        assert shell.port is None


class TestEveryCommandAnswers:
    def test_views_and_usage_errors(self, capsys) -> None:
        _, out = screen(
            capsys,
            "layers",
            "list layers",
            "filters",
            "list filters --inactive",
            "list nonsense",
            "preset",
            "clear",
            "add",
            "remove thing",
            "remove filter",
            "remove filter IBAN_X",
            "remove layer ner",
            "remove layer gpu",
            "add layer regex",
            "stop-serve",
            "/",
        )
        for expected in (
            "not set up",  # layers
            "Found by",  # filters
            "SSN",  # list filters --inactive
            "usage: list filters",
            "kvkk",  # preset list
            "usage: add filter",
            "usage: remove filter",
            "not an active filter",
            "already off",
            "usage: remove layer",
            "layer regex is already on",
            "not serving",
        ):
            assert expected in out, expected

    def test_no_filters_says_so(self, capsys) -> None:
        names = ",".join(
            [
                "CREDIT_CARD",
                "CUSTOM_SECRET",
                "EMAIL",
                "IBAN",
                "JWT",
                "PHONE",
                "TC_ID",
                "VEHICLE_PLATE",
            ]
        )
        _, out = screen(capsys, f"remove filter {names}", "filters", "scan a@b.io")
        assert "no filters are on" in out and "nothing found" in out


class TestPromptSession:
    def test_history_and_palette_of_the_real_prompt(self, capsys, state_home) -> None:
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.document import Document
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        from wardcat.cli.shell import _prompt_session

        shell, _ = screen(capsys)
        # A pipe and a dummy output, so this runs without a console on Windows too.
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            session = _prompt_session(shell)
        session.history.store_string(f"scan card {CARD}")
        session.history.store_string("add filter EMAIL")
        kept = (state_home / "wardcat" / "shell_history").read_text(encoding="utf-8")
        assert "add filter EMAIL" in kept and CARD not in kept
        if os.name == "posix":
            mode = (state_home / "wardcat" / "shell_history").stat().st_mode
            assert stat.S_IMODE(mode) == 0o600
        found = list(session.completer.get_completions(Document("/pre"), None))
        assert [c.text for c in found] == ["preset"]


class TestSweepFixes:
    def test_a_bad_action_names_the_choices(self, capsys) -> None:
        out = screen(capsys, "add filter TC_ID --action shred")[1]
        assert "unknown action 'shred'; choose from hash, mask, redact" in out
        assert "register_action" not in out

    @pytest.mark.parametrize("port", ["0", "70000"])
    def test_a_port_out_of_range(self, capsys, port) -> None:
        assert "between 1 and 65535" in screen(capsys, f"serve --port {port}")[1]

    def test_a_port_in_use_is_said_plainly(self, capsys) -> None:
        pytest.importorskip("uvicorn")
        import socket

        with socket.socket() as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen()
            port = taken.getsockname()[1]
            shell, out = screen(capsys, f"serve --port {port}")
        assert f"port {port} is in use" in out and "Traceback" not in out and shell.port is None

    def test_leaving_a_question_cancels_the_command(self, capsys, monkeypatch) -> None:
        def eof(prompt=""):
            raise EOFError

        monkeypatch.setattr("builtins.input", eof)
        args = argparse.Namespace(screen_resume=None, screen_continue=False, screen_salt_env=None)
        shell = Shell.start(args, interactive=False)
        shell.dispatch("add", "layer llm")
        shell.dispatch("scan", "")
        out = capsys.readouterr().out
        assert out.count("cancelled") == 2 and shell.state["llm"] is None

    def test_a_long_last_column_wraps_at_the_terminal(self, monkeypatch) -> None:
        import os as _os

        from wardcat.cli._shell_ui import table

        monkeypatch.setattr(
            "shutil.get_terminal_size", lambda fallback=None: _os.terminal_size((60, 24))
        )
        text = table(["Name", "Covers"], [["kvkk", "word " * 30]])
        assert all(len(line) <= 60 for line in text.splitlines())
        assert text.splitlines()[3].startswith(" " * 6)


class TestNerAsPeopleTypeIt:
    @pytest.mark.parametrize(
        ("words", "expected"),
        [
            (["tr"], ("tr", None)),
            (["turkish", "sm"], ("tr", "sm")),
            (["Türkçe", "md"], ("tr", "md")),
            (["english"], ("en", None)),
        ],
    )
    def test_language_and_size_as_words(self, words, expected) -> None:
        from wardcat.cli.shell import _ner_words

        assert _ner_words(words, None, None) == expected

    def test_flags_take_names_too(self) -> None:
        from wardcat.cli.shell import _ner_words

        assert _ner_words([], "turkish", "lg") == ("tr", "lg")

    @pytest.mark.parametrize("words", [["tr", "md", "lg"], ["md"]])
    def test_what_does_not_parse(self, words) -> None:
        from wardcat.cli.shell import _ner_words
        from wardcat.exceptions import ConfigError

        with pytest.raises(ConfigError):
            _ner_words(words, None, None)

    def test_ner_with_nothing_to_find_gets_the_person_types(self, capsys, monkeypatch) -> None:
        from wardcat import Wardcat

        monkeypatch.setattr(Wardcat, "with_ner", lambda self, **kw: self)
        shell, out = screen(capsys, "add layer ner turkish sm")
        assert shell.state["ner"] == {"language": "tr", "size": "sm"}
        assert {"PERSON", "ORG", "LOCATION"} <= set(shell.state["filters"])
        assert "no filter used the ner layer" in out

    def test_existing_person_filters_are_left_alone(self, capsys, monkeypatch) -> None:
        from wardcat import Wardcat

        monkeypatch.setattr(Wardcat, "with_ner", lambda self, **kw: self)
        shell, out = screen(capsys, "add filter PERSON=mask", "add layer ner tr")
        assert shell.state["filters"]["PERSON"] == "mask" and "ORG" not in shell.state["filters"]
        assert "no filter used" not in out

    def test_words_only_belong_to_ner(self, capsys) -> None:
        assert "unexpected 'tr'" in screen(capsys, "add layer regex tr")[1]


class TestWithoutTheNerLayerInstalled:
    def test_no_spacy_refuses_and_says_how_to_install(self, capsys, monkeypatch) -> None:
        import importlib.util

        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a: None if name == "spacy" else real(name, *a),
        )
        shell, out = screen(capsys, "add layer ner tr")
        assert 'pip install "wardcat[ner]"' in out and "wardcat models pull tr_core_news_md" in out
        assert "ner" not in shell.state["layers"] and "PERSON" not in shell.state["filters"]

    def test_a_missing_model_is_offered_and_can_be_declined(self, capsys, monkeypatch) -> None:
        from wardcat.ner import downloader

        monkeypatch.setattr(downloader, "is_installed", lambda name: False)
        monkeypatch.setattr("builtins.input", lambda prompt="": "n")
        shell, out = screen(capsys, "add layer ner tr")
        assert "tr_core_news_md model" in out and "wardcat models pull tr_core_news_md" in out
        assert "ner" not in shell.state["layers"]

    def test_a_missing_model_can_be_downloaded_from_the_screen(self, capsys, monkeypatch) -> None:
        from wardcat import Wardcat
        from wardcat.ner import downloader

        pulled = []
        monkeypatch.setattr(downloader, "is_installed", lambda name: False)
        monkeypatch.setattr(
            downloader, "download_model", lambda name, verbose=False: pulled.append(name)
        )
        monkeypatch.setattr(Wardcat, "with_ner", lambda self, **kw: self)
        monkeypatch.setattr("builtins.input", lambda prompt="": "evet")
        shell, _ = screen(capsys, "add layer ner tr")
        assert pulled == ["tr_core_news_md"] and "ner" in shell.state["layers"]

    def test_a_saved_ner_layer_without_its_model_opens_off(self, capsys, monkeypatch) -> None:
        from wardcat import Wardcat
        from wardcat.ner import downloader

        monkeypatch.setattr(Wardcat, "with_ner", lambda self, **kw: self)
        first, _ = screen(capsys, "add filter PERSON", "add layer ner tr")
        monkeypatch.setattr(downloader, "is_installed", lambda name: False)
        again, out = screen(capsys, resume=first.state["id"])
        assert again.state["layers"] == ["regex"] and "the ner layer is off" in out


class TestHints:
    def test_text_without_scan_points_at_scan(self, capsys) -> None:
        assert "start with scan: scan ahmet yılmaz" in screen(capsys, "ahmet yılmaz")[1]

    def test_short_words_for_filter_and_layer(self, capsys) -> None:
        shell, out = screen(capsys, "add fil IBAN=mask", "remove lay regex")
        assert shell.state["filters"]["IBAN"] == "mask" and "one layer must stay on" in out

    def test_nothing_found_without_a_model_layer_mentions_ner(self, capsys) -> None:
        assert "add layer ner tr" in screen(capsys, "scan ahmet yılmaz")[1]

"""The Claude Code hook, ``check --git-diff`` and shell completion."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from wardcat.cli import EXIT_CLEAN, EXIT_CONFIG, EXIT_FOUND, main
from wardcat.cli.check import parse_added_lines

CARD = "4111 1111 1111 1111"
POLICY = ["--entity", "EMAIL,CREDIT_CARD"]


def hook(capsys, monkeypatch, event, *extra: str):
    raw = event if isinstance(event, bytes) else json.dumps(event).encode()
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(raw)))
    code = main(["hook", "claude-code", *extra])
    out, err = capsys.readouterr()
    return code, (json.loads(out) if out.strip() else None), err


class TestHookPrompt:
    def test_a_prompt_with_a_finding_is_blocked_by_type(self, capsys, monkeypatch) -> None:
        event = {"hook_event_name": "UserPromptSubmit", "prompt_text": "mail ali@example.com"}
        code, out, _ = hook(capsys, monkeypatch, event, *POLICY)
        assert code == EXIT_CLEAN and out["decision"] == "block"
        assert "EMAIL" in out["reason"] and "ali@example.com" not in json.dumps(out)

    @pytest.mark.parametrize("field", ["prompt_text", "prompt"])
    def test_a_clean_prompt_passes_silently(self, capsys, monkeypatch, field) -> None:
        event = {"hook_event_name": "UserPromptSubmit", field: "hello"}
        assert hook(capsys, monkeypatch, event, *POLICY)[:2] == (EXIT_CLEAN, None)

    def test_no_prompt_text_blocks(self, capsys, monkeypatch) -> None:
        code, out, err = hook(capsys, monkeypatch, {"hook_event_name": "UserPromptSubmit"}, *POLICY)
        assert code == 2 and out is None and "blocked" in err


class TestHookToolCall:
    def test_arguments_with_a_finding_are_denied(self, capsys, monkeypatch) -> None:
        event = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Write",
            "tool_input": {"file_path": "a.txt", "content": f"card {CARD}"},
        }
        code, out, _ = hook(capsys, monkeypatch, event, *POLICY)
        decision = out["hookSpecificOutput"]
        assert code == EXIT_CLEAN and decision["permissionDecision"] == "deny"
        assert "CREDIT_CARD" in decision["permissionDecisionReason"]
        assert CARD not in json.dumps(out)

    def test_ask_instead_of_deny(self, capsys, monkeypatch) -> None:
        event = {"hook_event_name": "PreToolUse", "tool_input": {"command": "echo a@b.io"}}
        out = hook(capsys, monkeypatch, event, *POLICY, "--tool-decision", "ask")[1]
        assert out["hookSpecificOutput"]["permissionDecision"] == "ask"

    def test_clean_arguments_pass(self, capsys, monkeypatch) -> None:
        event = {"hook_event_name": "PreToolUse", "tool_input": {"command": "ls", "n": [1, "x"]}}
        assert hook(capsys, monkeypatch, event, *POLICY)[:2] == (EXIT_CLEAN, None)


class TestHookToolOutput:
    def test_text_output_is_sanitized(self, capsys, monkeypatch) -> None:
        event = {
            "hook_event_name": "PostToolUse",
            "tool_name": "Read",
            "tool_response": {"type": "text", "text": "to: ali@example.com"},
        }
        out = hook(capsys, monkeypatch, event, *POLICY)[1]["hookSpecificOutput"]
        assert out["updatedToolOutput"] == {"type": "text", "text": "to: [EMAIL]"}
        assert "EMAIL" in out["additionalContext"]

    def test_mcp_output_uses_its_own_key(self, capsys, monkeypatch) -> None:
        event = {
            "hook_event_name": "PostToolUse",
            "tool_name": "mcp__db__query",
            "tool_response": [{"row": "ali@example.com", "n": 1}],
        }
        out = hook(capsys, monkeypatch, event, *POLICY)[1]["hookSpecificOutput"]
        text = out["updatedMCPToolOutput"]["text"]
        assert json.loads(text) == [{"row": "[EMAIL]", "n": 1}]

    def test_clean_output_is_left_alone(self, capsys, monkeypatch) -> None:
        event = {"hook_event_name": "PostToolUse", "tool_response": "all good"}
        assert hook(capsys, monkeypatch, event, *POLICY)[:2] == (EXIT_CLEAN, None)

    def test_output_that_cannot_be_scanned_is_withheld(self, capsys, monkeypatch) -> None:
        event = {"hook_event_name": "PostToolUse", "tool_name": "Read", "tool_response": "x"}
        code, out, _ = hook(capsys, monkeypatch, event)  # no policy: the guard cannot be built
        text = out["hookSpecificOutput"]["updatedToolOutput"]["text"]
        assert code == EXIT_CLEAN and text.startswith("[wardcat withheld this output")


class TestHookFailsClosed:
    def test_unreadable_input_blocks(self, capsys, monkeypatch) -> None:
        assert hook(capsys, monkeypatch, b"not json", *POLICY)[0] == 2

    def test_a_scan_error_blocks_and_names_only_the_type(self, capsys, monkeypatch) -> None:
        from wardcat import Wardcat

        def broken(self, text):
            raise RuntimeError(f"leaks {text}")

        monkeypatch.setattr(Wardcat, "scan", broken)
        event = {"hook_event_name": "PreToolUse", "tool_input": {"command": "a@b.io"}}
        code, _, err = hook(capsys, monkeypatch, event, *POLICY)
        assert code == 2 and "RuntimeError" in err and "a@b.io" not in err

    def test_an_event_it_does_not_handle_is_a_visible_non_blocking_error(
        self, capsys, monkeypatch
    ) -> None:
        # exit 2 on Stop would mean "keep going", so it must not be used here
        code, out, err = hook(capsys, monkeypatch, {"hook_event_name": "Stop"}, *POLICY)
        assert (code, out) == (EXIT_FOUND, None) and "Stop" in err


class TestParseAddedLines:
    def test_numbers_follow_the_new_file(self) -> None:
        diff = (
            "diff --git a/f.txt b/f.txt\n--- a/f.txt\n+++ b/f.txt\n"
            "@@ -1,0 +2 @@\n+added two\n"
            "@@ -3 +5,2 @@\n-old\n+five\n+six\n"
        )
        assert list(parse_added_lines(diff)) == [
            ("f.txt", [2, 5, 6], ["added two\n", "five\n", "six\n"])
        ]

    def test_an_added_line_starting_with_plus_plus_is_content(self) -> None:
        diff = "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -0,0 +1,2 @@\n+++ not a header\n+x\n"
        assert list(parse_added_lines(diff)) == [("f", [1, 2], ["++ not a header\n", "x\n"])]

    def test_quoted_and_binary_paths(self) -> None:
        diff = (
            'diff --git "a/caf\\303\\251 x.txt" "b/caf\\303\\251 x.txt"\n'
            '--- /dev/null\n+++ "b/caf\\303\\251 x.txt"\n@@ -0,0 +1 @@\n+hi\n'
            "diff --git a/img.png b/img.png\nnew file mode 100644\n"
            "Binary files /dev/null and b/img.png differ\n"
        )
        assert list(parse_added_lines(diff)) == [
            ("café x.txt", [1], ["hi\n"]),
            ("img.png", [], None),
        ]


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
class TestGitDiff:
    @pytest.fixture
    def repo(self, tmp_path, monkeypatch) -> Path:
        def git(*a: str) -> None:
            subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)

        git("init", "-q")
        git("config", "user.email", "t@example.invalid")
        git("config", "user.name", "t")
        (tmp_path / "f.txt").write_text(f"a\ncard {CARD}\n", encoding="utf-8")
        git("add", ".")
        git("commit", "-qm", "init")
        (tmp_path / "f.txt").write_text(f"a\nmail ali@example.com\ncard {CARD}\n", encoding="utf-8")
        git("add", ".")
        monkeypatch.chdir(tmp_path)
        return tmp_path

    def test_only_added_lines_at_their_real_numbers(self, repo, capsys) -> None:
        code = main(["check", "--git-diff", *POLICY, "--format", "jsonl"])
        rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert code == EXIT_FOUND
        assert [(r["file"], r["line"], r["entity_type"]) for r in rows] == [("f.txt", 2, "EMAIL")]

    def test_against_a_ref_and_narrowed_by_path(self, repo, capsys) -> None:
        assert main(["check", "--git-diff", "HEAD", "f.txt", *POLICY]) == EXIT_FOUND
        assert main(["check", "--git-diff", "HEAD", "other.txt", *POLICY]) == EXIT_CLEAN

    def test_a_bad_ref_is_an_error(self, repo, capsys) -> None:
        assert main(["check", "--git-diff", "no-such-ref", *POLICY]) == EXIT_CONFIG

    def test_outside_a_repository_is_an_error(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
        assert main(["check", "--git-diff", *POLICY]) == EXIT_CONFIG

    def test_paths_or_git_diff_are_required(self, capsys) -> None:
        assert main(["check", *POLICY]) == EXIT_CONFIG
        assert main(["check", "-", "--git-diff", *POLICY]) == EXIT_CONFIG


class TestCompletion:
    @pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
    def test_scripts_name_commands_and_choices(self, capsys, shell) -> None:
        assert main(["completion", shell]) == EXIT_CLEAN
        script = capsys.readouterr().out
        for word in ("restore", "claude-code", "sarif", "kvkk", "TC_ID"):
            assert word in script

    @pytest.mark.skipif(
        shutil.which("bash") is None or sys.platform == "win32", reason="needs a POSIX bash"
    )
    def test_bash_completes_an_option_value(self, capsys, tmp_path) -> None:
        main(["completion", "bash"])
        script = tmp_path / "c.bash"
        script.write_text(capsys.readouterr().out, encoding="utf-8")
        probe = (
            f"source {script}; COMP_WORDS=(wardcat check --format s); COMP_CWORD=3; "
            '_wardcat; echo "${COMPREPLY[@]}"'
        )
        done = subprocess.run(["bash", "-c", probe], capture_output=True, text=True, check=True)
        assert done.stdout.split() == ["sarif"]

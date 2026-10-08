"""The CLI reaches what the library offers: tuning flags, batches, the round trip.

Each command here closes a gap where a library feature had no way in from the
command line, and each keeps the CLI's rules: no value is printed that was not
asked for, and a text that could not be scanned is never written out.
"""

from __future__ import annotations

import io
import json
import os
import stat
import sys

import pytest

from wardcat import Wardcat
from wardcat.cli import EXIT_CLEAN, EXIT_CONFIG, EXIT_FOUND, main

CARD = "4111 1111 1111 1111"


def run(capsys, *argv, stdin: str | None = None, monkeypatch=None):
    if stdin is not None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    code = main([str(a) for a in argv])
    out, err = capsys.readouterr()
    return code, out, err


class TestTuningFlags:
    def test_min_confidence_drops_weaker_matches(self, capsys, monkeypatch) -> None:
        text = "Phone: 905-674-3793"  # a cued phone, 0.90
        assert (
            run(capsys, "scan", "--entity", "PHONE", stdin=text, monkeypatch=monkeypatch)[0]
            == EXIT_FOUND
        )
        code, out, _ = run(
            capsys,
            "scan",
            "--entity",
            "PHONE",
            "--min-confidence",
            "0.95",
            stdin=text,
            monkeypatch=monkeypatch,
        )
        assert code == EXIT_CLEAN and out == f"{text}\n"

    def test_min_confidence_out_of_range(self, capsys, monkeypatch) -> None:
        code = run(
            capsys,
            "scan",
            "--entity",
            "EMAIL",
            "--min-confidence",
            "2",
            stdin="x",
            monkeypatch=monkeypatch,
        )[0]
        assert code == EXIT_CONFIG

    def test_propagate(self, capsys, monkeypatch) -> None:
        seen = {}
        real = Wardcat.with_propagation

        def spy(self, *a, **kw):
            seen["called"] = True
            return real(self, *a, **kw)

        monkeypatch.setattr(Wardcat, "with_propagation", spy)
        run(capsys, "scan", "--entity", "EMAIL", "--propagate", stdin="x", monkeypatch=monkeypatch)
        assert seen == {"called": True}

    def test_locale_shapes_surrogates(self, capsys, monkeypatch) -> None:
        monkeypatch.setenv("S", "s" * 16)
        code, out, _ = run(
            capsys,
            "scan",
            "--entity",
            "TC_ID=surrogate",
            "--locale",
            "tr",
            "--salt-env",
            "S",
            stdin="TC 10000000146",
            monkeypatch=monkeypatch,
        )
        assert code == EXIT_FOUND and "10000000146" not in out and out.startswith("TC ")

    def test_ner_size_needs_a_language(self, capsys, monkeypatch) -> None:
        code = run(
            capsys,
            "scan",
            "--entity",
            "PERSON",
            "--ner-size",
            "lg",
            stdin="x",
            monkeypatch=monkeypatch,
        )[0]
        assert code == EXIT_CONFIG

    def test_llm_timeout_needs_an_llm(self, capsys, monkeypatch) -> None:
        code = run(
            capsys,
            "scan",
            "--entity",
            "EMAIL",
            "--llm-timeout",
            "5",
            stdin="x",
            monkeypatch=monkeypatch,
        )[0]
        assert code == EXIT_CONFIG

    def test_phone_region_reaches_the_guard(self, capsys, monkeypatch) -> None:
        seen = {}
        real = Wardcat.with_phone_regions

        def spy(self, *regions):
            seen["regions"] = regions
            return real(self, *regions)

        monkeypatch.setattr(Wardcat, "with_phone_regions", spy)
        run(
            capsys,
            "scan",
            "--entity",
            "PHONE",
            "--phone-region",
            "gb,us",
            "--phone-region",
            "ES",
            stdin="x",
            monkeypatch=monkeypatch,
        )
        assert seen["regions"] == ("GB", "US", "ES")


class TestBatches:
    def test_several_files_into_a_directory(self, tmp_path, capsys) -> None:
        a, b = tmp_path / "a.txt", tmp_path / "b.txt"
        a.write_text(f"card {CARD}", encoding="utf-8")
        b.write_text("nothing here", encoding="utf-8")
        out_dir = tmp_path / "clean"
        code, out, _ = run(capsys, "scan", a, b, "--entity", "CREDIT_CARD", "--output-dir", out_dir)
        assert code == EXIT_FOUND and out == ""
        assert (out_dir / "a.txt").read_text(encoding="utf-8") == "card [CREDIT_CARD]\n"
        assert (out_dir / "b.txt").read_text(encoding="utf-8") == "nothing here\n"

    def test_several_files_need_a_directory(self, tmp_path, capsys) -> None:
        a = tmp_path / "a.txt"
        a.write_text("x", encoding="utf-8")
        assert run(capsys, "scan", a, a, "--entity", "EMAIL")[0] == EXIT_CONFIG

    def test_jsonl_sanitizes_one_field_and_keeps_the_rest(self, capsys, monkeypatch) -> None:
        data = "\n".join(
            [
                json.dumps({"id": 1, "text": f"card {CARD}", "lang": "en"}),
                json.dumps({"id": 2, "text": "mail ali@example.com"}),
            ]
        )
        code, out, _ = run(
            capsys,
            "scan",
            "--jsonl",
            "--entity",
            "CREDIT_CARD,EMAIL",
            stdin=data,
            monkeypatch=monkeypatch,
        )
        rows = [json.loads(line) for line in out.splitlines()]
        assert code == EXIT_FOUND
        assert rows == [
            {"id": 1, "text": "card [CREDIT_CARD]", "lang": "en"},
            {"id": 2, "text": "mail [EMAIL]"},
        ]

    def test_jsonl_with_another_field(self, capsys, monkeypatch) -> None:
        data = json.dumps({"prompt": "mail ali@example.com"})
        code, out, _ = run(
            capsys,
            "scan",
            "--jsonl",
            "--field",
            "prompt",
            "--entity",
            "EMAIL",
            stdin=data,
            monkeypatch=monkeypatch,
        )
        assert json.loads(out) == {"prompt": "mail [EMAIL]"}

    @pytest.mark.parametrize("line", ["not json", "[1]", '{"other": "x"}', '{"text": 5}'])
    def test_a_bad_jsonl_line_is_an_error(self, capsys, monkeypatch, line) -> None:
        code, out, err = run(
            capsys, "scan", "--jsonl", "--entity", "EMAIL", stdin=line, monkeypatch=monkeypatch
        )
        assert code == EXIT_CONFIG and out == "" and "line 1" in err

    def test_an_unscannable_item_writes_nothing(self, tmp_path, capsys, monkeypatch) -> None:
        policy = tmp_path / "p.yaml"
        policy.write_text(
            "max_text_bytes: 50\nentities:\n  EMAIL: {enabled: true, action: redact}\n",
            encoding="utf-8",
        )
        data = "\n".join(
            [json.dumps({"text": "a@b.io"}), json.dumps({"text": "x" * 500 + " a@b.io"})]
        )
        code, out, err = run(
            capsys, "scan", "--jsonl", "--config", policy, stdin=data, monkeypatch=monkeypatch
        )
        assert code == EXIT_CONFIG and out == "" and "nothing was written" in err


class TestRoundTrip:
    def test_tokenize_then_restore(self, tmp_path, capsys, monkeypatch) -> None:
        token_map = tmp_path / "map.json"
        code, prompt, _ = run(
            capsys,
            "scan",
            "--entity",
            "EMAIL=tokenize,CREDIT_CARD=tokenize",
            "--token-map",
            token_map,
            stdin=f"mail ali@example.com about card {CARD}",
            monkeypatch=monkeypatch,
        )
        assert code == EXIT_FOUND and "ali@example.com" not in prompt
        answer = prompt.replace("mail", "I wrote to").strip()
        code, restored, _ = run(
            capsys, "restore", "--token-map", token_map, stdin=answer, monkeypatch=monkeypatch
        )
        assert code == EXIT_CLEAN
        assert restored == f"I wrote to ali@example.com about card {CARD}\n"

    @pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
    def test_the_map_is_owner_only_even_over_an_existing_file(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        token_map = tmp_path / "map.json"
        token_map.write_text("{}", encoding="utf-8")
        token_map.chmod(0o644)
        run(
            capsys,
            "scan",
            "--entity",
            "EMAIL=tokenize",
            "--token-map",
            token_map,
            stdin="a@b.io",
            monkeypatch=monkeypatch,
        )
        assert stat.S_IMODE(token_map.stat().st_mode) == 0o600

    def test_no_map_unless_asked(self, tmp_path, capsys, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        run(capsys, "scan", "--entity", "EMAIL=tokenize", stdin="a@b.io", monkeypatch=monkeypatch)
        assert list(tmp_path.iterdir()) == []

    def test_a_foreign_placeholder_is_left_and_reported(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        token_map = tmp_path / "map.json"
        _, prompt, _ = run(
            capsys,
            "scan",
            "--entity",
            "EMAIL=tokenize",
            "--token-map",
            token_map,
            stdin="a@b.io",
            monkeypatch=monkeypatch,
        )
        answer = prompt.strip() + " and [EMAIL_1_ffffffffffff]"
        code, out, err = run(
            capsys, "restore", "--token-map", token_map, stdin=answer, monkeypatch=monkeypatch
        )
        assert code == EXIT_FOUND
        assert out == "a@b.io and [EMAIL_1_ffffffffffff]\n"
        assert "foreign" in err and "a@b.io" not in err

    def test_strict_refuses_without_printing(self, tmp_path, capsys, monkeypatch) -> None:
        token_map = tmp_path / "map.json"
        run(
            capsys,
            "scan",
            "--entity",
            "EMAIL=tokenize",
            "--token-map",
            token_map,
            stdin="a@b.io",
            monkeypatch=monkeypatch,
        )
        code, out, _ = run(
            capsys,
            "restore",
            "--token-map",
            token_map,
            "--strict",
            stdin="[EMAIL_1_ffffffffffff]",
            monkeypatch=monkeypatch,
        )
        assert code == EXIT_CONFIG and out == ""

    def test_not_a_token_map(self, tmp_path, capsys, monkeypatch) -> None:
        bad = tmp_path / "x.json"
        bad.write_text("{}", encoding="utf-8")
        assert (
            run(capsys, "restore", "--token-map", bad, stdin="x", monkeypatch=monkeypatch)[0]
            == EXIT_CONFIG
        )


class TestClassify:
    def test_categories_without_the_reason(self, capsys, monkeypatch) -> None:
        from wardcat.core.models import SensitivityVerdict

        def fake(self, text):
            return SensitivityVerdict(True, ("health", "pii"), reason=f"quotes: {text}")

        monkeypatch.setattr(Wardcat, "classify", fake)
        code, out, _ = run(
            capsys,
            "is-sensitive",
            "--llm",
            "m",
            "--categories",
            stdin="Ali is in rehab",
            monkeypatch=monkeypatch,
        )
        assert code == EXIT_FOUND
        assert json.loads(out) == {"sensitive": True, "categories": ["health", "pii"]}
        assert "rehab" not in out


class TestPresetsAndModels:
    def test_presets_list(self, capsys) -> None:
        code, out, _ = run(capsys, "presets")
        assert code == EXIT_CLEAN and {line.split()[0] for line in out.splitlines()} >= {
            "kvkk",
            "gdpr",
        }

    def test_preset_detail(self, capsys) -> None:
        code, out, _ = run(capsys, "presets", "kvkk")
        assert code == EXIT_CLEAN and "TC_ID" in out and "not covered:" in out

    def test_unknown_preset(self, capsys) -> None:
        assert run(capsys, "presets", "nope")[0] == EXIT_CONFIG

    def test_models_list(self, capsys) -> None:
        code, out, err = run(capsys, "models", "list", "--language", "de")
        assert code == EXIT_CLEAN and "de_core_news_sm" in out and "no NER component" in out
        assert "W094" not in err

    def test_pull_refuses_names_outside_the_catalog(self, capsys) -> None:
        assert run(capsys, "models", "pull", "evil-package")[0] == EXIT_CONFIG

    def test_pull_refuses_a_model_without_ner(self, capsys) -> None:
        assert run(capsys, "models", "pull", "de_dep_news_trf")[0] == EXIT_CONFIG

    def test_pull_installs_through_the_downloader(self, capsys, monkeypatch) -> None:
        from wardcat.ner import downloader

        calls = []
        monkeypatch.setattr(downloader, "is_installed", lambda name: False)
        monkeypatch.setattr(
            downloader, "download_model", lambda name, verbose=False: calls.append(name)
        )
        assert run(capsys, "models", "pull", "de_core_news_sm")[0] == EXIT_CLEAN
        assert calls == ["de_core_news_sm"]


class TestCheckJobs:
    def test_jobs_give_the_same_findings(self, tmp_path, capsys) -> None:
        for i in range(6):
            (tmp_path / f"f{i}.txt").write_text(
                f"line\ncard {CARD} mail a{i}@b.io\n", encoding="utf-8"
            )
        _, one, _ = run(capsys, "check", tmp_path, "--entity", "CREDIT_CARD,EMAIL")
        code, many, _ = run(
            capsys, "check", tmp_path, "--entity", "CREDIT_CARD,EMAIL", "--jobs", "3"
        )
        assert code == EXIT_FOUND and sorted(one.splitlines()) == sorted(many.splitlines())

    def test_jobs_must_be_positive(self, tmp_path, capsys) -> None:
        assert run(capsys, "check", tmp_path, "--jobs", "0")[0] == EXIT_CONFIG

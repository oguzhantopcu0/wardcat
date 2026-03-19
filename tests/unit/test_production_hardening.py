"""
Production sertleştirme testleri.

Kapsam:
- ReDoS (Regex Denial-of-Service) koruması
- Büyük input işleme
- Thread-safety (concurrent scan)
- scan_batch hata izolasyonu
- Ortam değişkeni konfigürasyonu
- Config validasyon
- Entity tip uyarısı
- Salt uyarısı
- Encoding hata yönetimi (CLI)
- __version__ export
"""
from __future__ import annotations

import logging
import os
import threading
import time
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_guard import LLMGuard, __version__
from ai_guard.config.loader import load_config, validate_config
from ai_guard.core.models import KNOWN_ENTITY_TYPES, warn_unknown_entity
from ai_guard.detectors.regex_detector import RegexDetector


# ── __version__ ───────────────────────────────────────────────────────────────

def test_version_exported():
    assert __version__ is not None
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


# ── ReDoS Koruması ────────────────────────────────────────────────────────────

class TestReDoS:
    """
    Regex engine'in yoğun/özel-yapılı input'larda kabul edilebilir sürede
    tamamlandığını doğrular.  pytest-timeout (pyproject.toml: timeout=30)
    ile birlikte çalışır — regex takılırsa test 30s'de kesilir.
    """

    def _detector(self):
        return RegexDetector({"EMAIL", "CREDIT_CARD", "PHONE", "IBAN", "TC_ID", "IP_ADDRESS"})

    def test_long_repetitive_input(self):
        """10 KB tekrarlayan karakter regex'i dondurmamalı."""
        text = "a" * 10_000
        t0 = time.perf_counter()
        self._detector().detect(text)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"Regex çok yavaş: {elapsed:.2f}s"

    def test_near_match_email_pattern(self):
        """E-posta'ya benzeyen ama eşleşmeyen dizi regex'i dondurmamalı."""
        text = ("a" * 50 + "@") * 200  # çok sayıda yanlış @
        t0 = time.perf_counter()
        self._detector().detect(text)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"Regex çok yavaş: {elapsed:.2f}s"

    def test_near_match_credit_card(self):
        """Kart numarasına benzeyen ama geçersiz dizi."""
        text = "4111 " * 2000  # tekrarlayan kart prefix'i
        t0 = time.perf_counter()
        self._detector().detect(text)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"Regex çok yavaş: {elapsed:.2f}s"

    def test_mixed_separators_iban(self):
        """IBAN formatına benzeyen ama hatalı uzun dizi."""
        text = "TR" + "0" * 500
        t0 = time.perf_counter()
        self._detector().detect(text)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"Regex çok yavaş: {elapsed:.2f}s"


# ── Büyük Input ───────────────────────────────────────────────────────────────

class TestLargeInput:
    def _guard(self):
        return LLMGuard(use_ner=False)

    def test_100kb_clean_text(self):
        """100 KB temiz metin işlenebilmeli."""
        text = "Bu temiz bir metin, hassas veri yok. " * 2_800  # ~100KB
        result = self._guard().scan(text)
        assert result.is_clean

    def test_100kb_with_pii(self):
        """100 KB metin içinde PII bulunabilmeli."""
        filler = "Lorem ipsum dolor sit amet. " * 1_000
        text = filler + " email: test@example.com " + filler
        result = self._guard().scan(text)
        emails = [v for v in result.violations if v.entity_type == "EMAIL"]
        assert len(emails) >= 1

    def test_batch_many_items(self):
        """100 öğeli batch işlenebilmeli, hiçbiri çökmemeli."""
        texts = [f"Kullanıcı {i}: user{i}@example.com" for i in range(100)]
        results = self._guard().scan_batch(texts)
        assert len(results) == 100
        assert all(
            any(v.entity_type == "EMAIL" for v in r.violations)
            for r in results
        )

    def test_scan_batch_large_items(self):
        """10 KB'lık 10 öğeli batch performans kontrolü."""
        texts = [("temiz metin " * 500) for _ in range(10)]
        t0 = time.perf_counter()
        results = LLMGuard(use_ner=False).scan_batch(texts)
        elapsed = time.perf_counter() - t0
        assert len(results) == 10
        assert elapsed < 10.0, f"Batch çok yavaş: {elapsed:.2f}s"


# ── Thread-Safety ─────────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_scan_no_crash(self):
        """
        Aynı LLMGuard instance'ı eşzamanlı 10 thread'den çağrılabilmeli.
        Sonuçlar tutarlı olmalı, exception olmamalı.
        """
        guard = LLMGuard(use_ner=False)
        texts = [
            "email: test@example.com",
            "kart: 4111111111111111",
            "temiz metin",
            "TC: 12345678950",
            "IBAN TR33 0006 1005 1978 6457 8413 26",
        ]
        errors: list[Exception] = []
        results: list = [None] * (len(texts) * 2)

        def worker(idx: int, text: str):
            try:
                results[idx] = guard.scan(text)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(i, texts[i % len(texts)]))
            for i in range(len(texts) * 2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors, f"Thread hataları: {errors}"
        assert all(r is not None for r in results)

    def test_concurrent_scan_batch(self):
        """scan_batch eşzamanlı çağrılabilmeli."""
        guard = LLMGuard(use_ner=False)
        batch = ["a@b.com", "temiz", "4111111111111111"] * 10
        errors: list[Exception] = []

        def worker():
            try:
                guard.scan_batch(batch)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors


# ── scan_batch Hata İzolasyonu ────────────────────────────────────────────────

class TestScanBatchIsolation:
    def test_exception_in_one_item_does_not_fail_others(self):
        """
        Bir öğede beklenmeyen hata olsa bile diğerleri başarıyla taranmalı.
        """
        guard = LLMGuard(use_ner=False)

        call_count = [0]
        original_scan = guard._engine.scan

        def flaky_scan(text):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simüle edilmiş hata")
            return original_scan(text)

        guard._engine.scan = flaky_scan

        texts = ["a@b.com", "4111111111111111", "c@d.com"]
        results = guard.scan_batch(texts)

        assert len(results) == 3
        # 2. öğe hata → orijinal metin döner, temiz sayılır
        assert results[1].original_text == "4111111111111111"
        assert results[1].is_clean
        # Diğerleri normal
        assert any(v.entity_type == "EMAIL" for v in results[0].violations)
        assert any(v.entity_type == "EMAIL" for v in results[2].violations)

    def test_empty_batch_returns_empty(self):
        assert LLMGuard(use_ner=False).scan_batch([]) == []


# ── Ortam Değişkeni Konfigürasyonu ────────────────────────────────────────────

class TestEnvVarConfig:
    def test_llmguard_salt_from_env(self):
        with patch.dict(os.environ, {"LLMGUARD_SALT": "env-test-salt"}):
            cfg = load_config()
        assert cfg["salt"] == "env-test-salt"

    def test_llmguard_llm_url_from_env(self):
        with patch.dict(os.environ, {"LLMGUARD_LLM_URL": "http://remote:11434"}):
            cfg = load_config()
        assert cfg["llm_detector"]["base_url"] == "http://remote:11434"

    def test_llmguard_llm_model_from_env(self):
        with patch.dict(os.environ, {"LLMGUARD_LLM_MODEL": "mistral:7b"}):
            cfg = load_config()
        assert cfg["llm_detector"]["model"] == "mistral:7b"

    def test_llmguard_spacy_model_from_env(self):
        with patch.dict(os.environ, {"LLMGUARD_SPACY_MODEL": "tr_core_news_sm"}):
            cfg = load_config()
        assert cfg["spacy_model"] == "tr_core_news_sm"

    def test_llmguard_timeout_from_env(self):
        with patch.dict(os.environ, {"LLMGUARD_LLM_TIMEOUT": "120"}):
            cfg = load_config()
        assert cfg["llm_detector"]["timeout"] == 120

    def test_invalid_timeout_env_ignored(self, caplog):
        with patch.dict(os.environ, {"LLMGUARD_LLM_TIMEOUT": "abc"}):
            with caplog.at_level(logging.WARNING, logger="ai_guard.config.loader"):
                cfg = load_config()
        # Geçersiz değer yok sayılır, default korunur
        assert cfg["llm_detector"]["timeout"] == 60

    def test_env_overrides_yaml(self, tmp_path):
        """Ortam değişkeni YAML değerinin üzerine yazmalı."""
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("salt: yaml-salt\n")
        with patch.dict(os.environ, {"LLMGUARD_SALT": "env-salt"}):
            cfg = load_config(yaml_file)
        assert cfg["salt"] == "env-salt"


# ── Config Validasyon ─────────────────────────────────────────────────────────

class TestConfigValidation:
    def test_invalid_action_raises(self):
        cfg = {"entities": {"EMAIL": {"enabled": True, "action": "invalid_action"}}}
        with pytest.raises(ValueError, match="Geçersiz action"):
            validate_config(cfg)

    def test_valid_actions_pass(self):
        cfg = {
            "entities": {
                "EMAIL": {"enabled": True, "action": "warn"},
                "TC_ID": {"enabled": True, "action": "hash"},
            },
            "llm_detector": {"backend": "ollama", "timeout": 60},
        }
        validate_config(cfg)  # hata fırlatmamalı

    def test_invalid_backend_raises(self):
        cfg = {
            "entities": {},
            "llm_detector": {"backend": "unknown_backend", "timeout": 60},
        }
        with pytest.raises(ValueError, match="Geçersiz LLM backend"):
            validate_config(cfg)

    def test_invalid_timeout_raises(self):
        cfg = {
            "entities": {},
            "llm_detector": {"backend": "ollama", "timeout": -5},
        }
        with pytest.raises(ValueError, match="Geçersiz.*timeout"):
            validate_config(cfg)

    def test_configure_entity_invalid_action_raises(self):
        guard = LLMGuard(use_ner=False)
        with pytest.raises(ValueError, match="Geçersiz action"):
            guard.configure_entity("EMAIL", action="delete")


# ── Entity Tip Uyarısı ────────────────────────────────────────────────────────

class TestEntityTypeWarning:
    def test_known_types_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="ai_guard.core.models"):
            for et in KNOWN_ENTITY_TYPES:
                warn_unknown_entity(et)
        assert "Bilinmeyen" not in caplog.text

    def test_typo_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="ai_guard.core.models"):
            warn_unknown_entity("CREIDT_CARD")  # typo
        assert "CREIDT_CARD" in caplog.text

    def test_configure_unknown_entity_warns(self, caplog):
        guard = LLMGuard(use_ner=False)
        with caplog.at_level(logging.WARNING, logger="ai_guard.core.models"):
            guard.configure_entity("TYPO_ENTITY", action="warn")
        assert "TYPO_ENTITY" in caplog.text


# ── Salt Uyarısı ──────────────────────────────────────────────────────────────

class TestSaltWarning:
    def test_empty_salt_with_hash_action_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="ai_guard.guard"):
            LLMGuard(use_ner=False, salt="")
        assert "LLMGUARD_SALT" in caplog.text

    def test_nonempty_salt_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="ai_guard.guard"):
            LLMGuard(use_ner=False, salt="my-secret-salt")
        assert "LLMGUARD_SALT" not in caplog.text


# ── CLI Encoding Hata Yönetimi ────────────────────────────────────────────────

class TestCLIEncodingError:
    def test_utf8_file_read_ok(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("ali@example.com", encoding="utf-8")
        from ai_guard.__main__ import _read_file
        content = _read_file(f)
        assert "ali@example.com" in content

    def test_nonexistent_file_raises(self, tmp_path):
        from ai_guard.__main__ import _read_file
        with pytest.raises(FileNotFoundError):
            _read_file(tmp_path / "yok.txt")

    def test_invalid_utf8_raises_valueerror(self, tmp_path):
        f = tmp_path / "bad.txt"
        f.write_bytes(b"\xff\xfe invalid utf-8 \x80\x81")
        from ai_guard.__main__ import _read_file
        with pytest.raises(ValueError, match="UTF-8"):
            _read_file(f)

    def test_cli_scan_invalid_utf8_exits_with_error(self, tmp_path):
        """cmd_scan → ValueError; main() bunu yakalayıp sys.exit(1) çağırır."""
        f = tmp_path / "bad.txt"
        f.write_bytes(b"\xff\xfe bad encoding")
        from ai_guard.__main__ import _build_parser, cmd_scan
        parser = _build_parser()
        args = parser.parse_args(["scan", "--file", str(f), "--no-ner"])
        # cmd_scan doğrudan ValueError fırlatır (main() onu yakalar, sys.exit(1) yapar)
        with pytest.raises(ValueError, match="UTF-8"):
            cmd_scan(args)

    def test_cli_scan_missing_file_raises_file_not_found(self, tmp_path):
        from ai_guard.__main__ import _build_parser, cmd_scan
        parser = _build_parser()
        args = parser.parse_args(["scan", "--file", str(tmp_path / "missing.txt"), "--no-ner"])
        with pytest.raises(FileNotFoundError):
            cmd_scan(args)


# ── Logging Entegrasyonu ──────────────────────────────────────────────────────

class TestLogging:
    def test_engine_logs_scan_completion(self, caplog):
        from ai_guard.core.engine import DetectionEngine
        from ai_guard.config.loader import load_config
        cfg = load_config()
        cfg["entities"]["EMAIL"]["enabled"] = True
        from ai_guard.detectors.regex_detector import RegexDetector
        engine = DetectionEngine(cfg, [RegexDetector({"EMAIL"})])
        with caplog.at_level(logging.INFO, logger="ai_guard.core.engine"):
            engine.scan("test@example.com")
        assert "scan tamamlandı" in caplog.text

    def test_llm_detector_logs_on_connection_error(self, caplog):
        from ai_guard.detectors.llm_detector import LLMDetector
        from unittest.mock import MagicMock
        backend = MagicMock()
        backend.complete_messages.side_effect = ConnectionError("bağlantı yok")
        det = LLMDetector(backend=backend, enabled_entities={"EMAIL"})
        with caplog.at_level(logging.WARNING, logger="ai_guard.detectors.llm_detector"):
            result = det.detect("test@example.com")
        assert result == []
        assert "bağlantı hatası" in caplog.text.lower()

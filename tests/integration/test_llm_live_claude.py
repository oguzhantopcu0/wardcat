"""
Claude API backend ile canlı LLM dedektör entegrasyon testleri.

Model indirmeye GEREK YOKTUR — Anthropic Claude API kullanılır.
ANTHROPIC_API_KEY ortam değişkeni gerektirir; yoksa testler atlanır.

Test hedefi:
  build_prompt() → ClaudeBackend.complete() → _parse_llm_response()
  → _locate_spans() → DetectionEngine → ScanResult (violations + hash)

Çalıştırma:
  ANTHROPIC_API_KEY=sk-... pytest tests/integration/test_llm_live_claude.py -v
"""
from __future__ import annotations

import os
import pytest

# ANTHROPIC_API_KEY yoksa tüm testleri atla
pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY ortam değişkeni tanımlı değil",
)


# ---------------------------------------------------------------------------
# Yardımcı: ClaudeBackend kullanan LLMGuard fabrikası
# ---------------------------------------------------------------------------

def _make_guard(entities: set[str] | None = None):
    """
    ClaudeBackend enjekte edilmiş LLMGuard döndürür.
    Regex dedektörü aktif tutulur (hibrit mod).
    """
    from ai_guard import LLMGuard
    from ai_guard.detectors.llm_detector import LLMDetector
    from ai_guard.core.engine import DetectionEngine
    from ai_guard.llm.backends.claude import ClaudeBackend

    enabled = entities or {
        "CREDIT_CARD", "EMAIL", "PERSON", "TC_ID",
        "IBAN", "PHONE", "IP_ADDRESS", "ADDRESS", "CUSTOM_SECRET",
    }

    guard = LLMGuard(use_ner=False, use_llm=False)  # sadece regex ile başlat

    # Engine config'e LLM entity'lerini ekle
    for e in enabled:
        guard._config["entities"].setdefault(e, {"enabled": True, "action": "warn"})
    guard._config["entities"]["PERSON"]        = {"enabled": True, "action": "hash"}
    guard._config["entities"]["CUSTOM_SECRET"] = {"enabled": True, "action": "hash"}
    guard._config["entities"]["CREDIT_CARD"]   = {"enabled": True, "action": "hash"}

    llm_det = LLMDetector(
        backend=ClaudeBackend(model="claude-haiku-4-5"),  # hızlı + ucuz
        enabled_entities=enabled,
    )
    guard._detectors.append(llm_det)
    guard._engine = DetectionEngine(guard._config, guard._detectors)
    return guard


# ---------------------------------------------------------------------------
# Tekil tespit — tam pipeline
# ---------------------------------------------------------------------------

class TestClaudeSingleDetection:
    def test_person_detected_and_hashed(self):
        """Claude kişi adını tespit edip hash'lemeli."""
        guard = _make_guard()
        text  = "Customer name: John Smith, please process the order."
        result = guard.scan(text)

        persons = [v for v in result.violations if v.entity_type == "PERSON"]
        assert len(persons) >= 1, "Claude PERSON'ı tespit edemedi"
        # Hash action uygulanmalı
        assert "John Smith" not in result.sanitized_text
        assert "[PERSON:" in result.sanitized_text

    def test_email_detected(self):
        """Claude e-posta adresini tespit etmeli."""
        guard = _make_guard()
        text  = "Please reach out to alice@example.com for more details."
        result = guard.scan(text)

        emails = [v for v in result.violations if v.entity_type == "EMAIL"]
        assert len(emails) >= 1, "EMAIL tespit edilemedi"

    def test_clean_text_returns_clean(self):
        """Temiz metin → ihlal olmamalı."""
        guard = _make_guard()
        result = guard.scan("The sky is blue and the weather is nice today.")
        assert result.is_clean, f"Beklenmedik ihlaller: {result.violations}"

    def test_span_positions_match_original(self):
        """LLM'in döndürdüğü span pozisyonları orijinal metinle örtüşmeli."""
        guard = _make_guard()
        name  = "Maria Garcia"
        text  = f"Hello, my name is {name} and I live in Madrid."
        result = guard.scan(text)

        for v in result.violations:
            if v.entity_type == "PERSON":
                assert text[v.start:v.end] == v.original, (
                    f"Pozisyon uyuşmazlığı: text[{v.start}:{v.end}]="
                    f"'{text[v.start:v.end]}' != '{v.original}'"
                )


# ---------------------------------------------------------------------------
# Hibrit mod: Regex + LLM birlikte
# ---------------------------------------------------------------------------

class TestClaudeHybridDetection:
    def test_regex_catches_card_llm_catches_person(self):
        """Regex kredi kartını, Claude kişiyi yakalamalı."""
        guard = _make_guard()
        text  = "Ali Veli paid with card 4111111111111111."
        result = guard.scan(text)

        entity_types = {v.entity_type for v in result.violations}
        assert "CREDIT_CARD" in entity_types, "Regex CREDIT_CARD'ı kaçırdı"
        assert "PERSON" in entity_types, "Claude PERSON'ı kaçırdı"

    def test_no_duplicate_for_same_span(self):
        """Email hem regex hem LLM tarafından yakalanabilir; tekrar olmamalı."""
        guard = _make_guard({"EMAIL"})
        # Email entity için action'ı warn olarak ayarla
        guard._config["entities"]["EMAIL"] = {"enabled": True, "action": "warn"}
        from ai_guard.core.engine import DetectionEngine
        guard._engine = DetectionEngine(guard._config, guard._detectors)

        result = guard.scan("Contact us at test@example.com")
        emails = [v for v in result.violations if v.entity_type == "EMAIL"]
        assert len(emails) == 1, f"Tekrar tespit: {len(emails)} EMAIL ihlali"


# ---------------------------------------------------------------------------
# Sanitized metin bütünlüğü
# ---------------------------------------------------------------------------

class TestClaudeSanitizedText:
    def test_person_replaced_in_sanitized(self):
        """Hash'lenen kişi adı sanitized_text'te bulunmamalı."""
        guard = _make_guard()
        name  = "Robert Johnson"
        text  = f"The account belongs to {name}."
        result = guard.scan(text)

        if any(v.entity_type == "PERSON" for v in result.violations):
            assert name not in result.sanitized_text
            assert "[PERSON:" in result.sanitized_text

    def test_multiple_entities_all_sanitized(self):
        """Birden fazla entity türü aynı anda doğru sanitize edilmeli."""
        guard = _make_guard()
        text  = "Send invoice to Jane Doe at jane@corp.com"
        result = guard.scan(text)

        # En az bir ihlal olmalı (PERSON veya EMAIL)
        assert not result.is_clean, "Metin tamamen temiz görünüyor"

        # Orijinal veriler sanitized_text'te olmamalı (hash/warn uygulananlar)
        for v in result.violations:
            if v.action.name == "HASH":
                assert v.original not in result.sanitized_text, (
                    f"'{v.original}' hâlâ sanitized_text'te var"
                )


# ---------------------------------------------------------------------------
# Hata toleransı — diğer testlerden bağımsız
# ---------------------------------------------------------------------------

class TestClaudeErrorTolerance:
    def test_regex_works_when_llm_present(self):
        """LLM dedektörü aktifken regex hâlâ çalışmalı."""
        guard = _make_guard()
        result = guard.scan("kart: 4111111111111111")
        assert any(v.entity_type == "CREDIT_CARD" for v in result.violations), (
            "Regex CREDIT_CARD'ı tespit edemedi (LLM aktifken)"
        )

    def test_tc_id_regex_independent(self):
        """TC kimlik numarası regex tarafından yakalanmalı."""
        guard = _make_guard()
        result = guard.scan("TC: 12345678901")
        assert any(v.entity_type == "TC_ID" for v in result.violations)


# ---------------------------------------------------------------------------
# Prompt → Claude → Parse pipeline doğrulama
# ---------------------------------------------------------------------------

class TestClaudePipelineIntegrity:
    def test_build_prompt_sent_to_claude(self):
        """
        build_prompt() çıktısının Claude'a iletildiğini ve
        geçerli JSON döndürdüğünü doğrular — LLMDetector iç akışını test eder.
        """
        from ai_guard.llm.backends.claude import ClaudeBackend
        from ai_guard.llm.prompt import build_prompt

        backend = ClaudeBackend(model="claude-haiku-4-5")
        prompt  = build_prompt("hello@world.com is my email", {"EMAIL"})

        raw = backend.complete(prompt)

        # Ham yanıt boş olmamalı
        assert isinstance(raw, str)
        assert len(raw.strip()) > 0

        # LLMDetector'ın parse mantığını manuel çalıştır
        import json, re  # noqa: E401
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        assert match is not None, f"JSON array bulunamadı:\n{raw}"
        parsed = json.loads(match.group())
        assert isinstance(parsed, list)

    def test_llm_detector_detect_method_directly(self):
        """LLMDetector.detect() doğrudan çağrıldığında DetectedSpan döndürmeli."""
        from ai_guard.detectors.llm_detector import LLMDetector
        from ai_guard.llm.backends.claude import ClaudeBackend

        det = LLMDetector(
            backend=ClaudeBackend(model="claude-haiku-4-5"),
            enabled_entities={"EMAIL"},
        )
        spans = det.detect("My email is user@example.com")

        emails = [s for s in spans if s.entity_type == "EMAIL"]
        assert len(emails) >= 1, "LLMDetector EMAIL'i tespit edemedi"
        assert emails[0].text == "user@example.com"
        assert emails[0].start >= 0
        assert emails[0].end > emails[0].start

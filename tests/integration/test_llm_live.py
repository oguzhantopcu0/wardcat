"""
Live LLM integration tests — exercise the full pipeline against a REAL model.

Unlike the mocked unit tests, these actually call an Ollama backend, so they
verify that the prompt works, the model returns parseable output, and PII is
detected end-to-end.

Marked ``slow`` and auto-skipped when Ollama is unreachable or no model is
installed — so a normal ``pytest`` run (and CI without Ollama) is unaffected.

Run explicitly:
    uv run pytest -m slow tests/integration/test_llm_live.py
Pick the model:
    AIGUARD_TEST_LLM_MODEL=llama3.2:1b uv run pytest -m slow ...

Non-determinism note: assertions target only unambiguous PII (full names,
emails, card numbers, explicitly labeled secrets) detected at temperature 0.
"""

from __future__ import annotations

import os

import pytest

from ai_guard import LLMGuard
from ai_guard.llm.backends.ollama import OllamaBackend

pytestmark = pytest.mark.slow

_OLLAMA_URL = os.environ.get("AIGUARD_TEST_OLLAMA_URL", "http://localhost:11434")

# Smallest-first preference — CI can install a tiny model; locally we use
# whatever is present. Matched by prefix so quantized tags (…-instruct-qX) hit.
_PREFERRED = [
    "llama3.2:1b",
    "qwen2.5:0.5b",
    "qwen2.5:3b",
    "llama3.2:3b",
    "gemma3:12b",
    "llama3.1:8b",
    "qwen2.5:7b",
]


def _pick_model() -> str | None:
    """Return an installed Ollama model to test with, or None to skip."""
    try:
        models = OllamaBackend(base_url=_OLLAMA_URL, model="none").list_models()
    except Exception:
        return None
    if not models:
        return None
    override = os.environ.get("AIGUARD_TEST_LLM_MODEL")
    if override:
        return override if override in models else None
    for pref in _PREFERRED:
        for m in models:
            if m == pref or m.startswith(pref):
                return m
    return models[0]


_MODEL = _pick_model()

needs_ollama = pytest.mark.skipif(
    _MODEL is None,
    reason="Ollama not reachable or no model installed (set AIGUARD_TEST_LLM_MODEL to choose).",
)


@pytest.fixture(scope="module")
def llm_guard() -> LLMGuard:
    """LLM-only guard (NER off) — so PERSON detection proves the LLM ran."""
    return LLMGuard(
        use_ner=False,
        use_llm=True,
        llm_model=_MODEL,
        llm_timeout=120,
        salt="live-test-salt",
    )


@needs_ollama
class TestLiveLLMDetection:
    def test_detects_person_only_llm_can_find(self, llm_guard):
        # NER is off, so a detected PERSON must have come from the LLM.
        result = llm_guard.scan("Müşterimiz Ali Veli ile görüşüldü, e-posta ali.veli@firma.com.")
        types = {v.entity_type for v in result.violations}
        assert "PERSON" in types, f"LLM missed PERSON; got {types}"
        assert "EMAIL" in types

    def test_detects_contextual_secret(self):
        # db_pass=VALUE has no known prefix → regex can't catch it; only the LLM can.
        guard = LLMGuard(use_ner=False, use_llm=True, llm_model=_MODEL, llm_timeout=120, salt="s")
        guard._config["llm_detector"]["entities"]["CUSTOM_SECRET"] = {
            "enabled": True,
            "action": "hash",
        }
        guard._rebuild()
        result = guard.scan("Veritabanı şifresi db_pass=S3cr3t!42 — kimseyle paylaşmayın.")
        types = {v.entity_type for v in result.violations}
        assert "CUSTOM_SECRET" in types, f"LLM missed contextual secret; got {types}"
        assert "S3cr3t!42" not in result.sanitized_text

    def test_clean_text_no_false_positive(self, llm_guard):
        result = llm_guard.scan("Bugün hava çok güzel, yarın pikniğe gideceğiz.")
        assert result.is_clean, f"false positives on clean text: {result.violations}"


@needs_ollama
class TestLiveAdjudication:
    def test_deterministic_kept_and_pipeline_runs(self):
        # Full stack (regex + NER + LLM adjudication). spaCy is required here.
        pytest.importorskip("spacy")
        guard = LLMGuard(
            use_ner=True,
            spacy_model="en_core_web_sm",
            use_llm=True,
            llm_model=_MODEL,
            llm_timeout=120,
            llm_adjudicate=True,
            salt="adj",
        )
        result = guard.scan("John Smith paid with card 4111 1111 1111 1111.")
        types = {v.entity_type for v in result.violations}
        # The Luhn-valid card is a deterministic regex span — must survive adjudication.
        assert "CREDIT_CARD" in types
        assert "4111 1111 1111 1111" not in result.sanitized_text

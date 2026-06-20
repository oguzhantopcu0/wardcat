"""
Hybrid detection: regex + SpaCy NER + on-prem LLM, with ensemble adjudication.

The LLM verifies the regex/NER candidates (dropping false positives, relabeling
mistakes) and adds contextual PII the other layers miss — all in one call.

Requires:
    pip install "ai-guard[ner]"
    ollama pull gemma3:12b   # or any chat model; set the name below
"""

from ai_guard import AIGuard

TEXT = """\
Müşteri Ali Veli (ali.veli@firma.com) ile görüşüldü.
Veritabanı şifresi db_pass=S3cr3t!42 loglarda açık kalmış.
Kredi kartı 4111 1111 1111 1111 ile ödeme yapıldı.
"""


def main() -> None:
    guard = AIGuard(
        use_ner=True,
        spacy_model="en_core_web_sm",
        use_llm=True,
        llm_backend="ollama",
        llm_model="gemma3:12b",
        llm_adjudicate=True,  # LLM acts as detector + arbiter in one call
        salt="example-salt",
    )
    # db_pass=... has no known prefix → only the LLM can flag it. Target the
    # LLM layer explicitly so it is not also enabled for regex.
    guard.add_entity("CUSTOM_SECRET", action="hash", layers=["llm"])

    result = guard.scan(TEXT)
    print("Sanitized:\n" + result.sanitized_text)
    print("\nViolations:")
    for v in result.violations:
        print(f"  [{v.action.value:6}] {v.entity_type:14} {v.original!r}")


if __name__ == "__main__":
    main()

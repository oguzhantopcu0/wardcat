"""
ai-guard kullanım örnekleri.
"""

from ai_guard import AIGuard

SAMPLE = """
Merhaba, ben Ahmet Yılmaz. Şirketimizin sunucu IP'si 10.0.0.42.
Bana fatih.demir@firma.com adresinden veya 0533 987 65 43 numarasından ulaşabilirsin.
Ödeme için IBAN: TR330006100519786457841326 kullanabilirsiniz.
Kredi kartım: 4532015112830366. TC kimliğim: 10987654321.
""".strip()


def demo_programmatic_api():
    print("=" * 60)
    print("Programmatic API")
    print("=" * 60)

    guard = (
        AIGuard(use_ner=False, salt="gizli-tuz-123")
        .add_entity("EMAIL", enabled=True, action="warn")
        .add_entity("CREDIT_CARD", enabled=True, action="hash")
        .add_entity("IBAN", enabled=True, action="hash")
        .add_entity("TC_ID", enabled=True, action="hash")
    )

    result = guard.scan(SAMPLE)

    print(f"\nTemizlenmiş metin:\n{result.sanitized_text}")
    print(f"\nToplam ihlal: {len(result.violations)}")
    for v in result.violations:
        if v.action == "hash":
            print(f"  [{v.action}] {v.entity_type}: '{v.original}' → '{v.replacement}'")
        else:
            print(f"  [{v.action}] {v.entity_type}: '{v.original}'")


def demo_yaml_api():
    print("\n" + "=" * 60)
    print("Declarative (YAML) API")
    print("=" * 60)

    guard = AIGuard(config_path="config/default.yaml", use_ner=False)
    result = guard.scan(SAMPLE)

    print(f"\nTemizlenmiş metin:\n{result.sanitized_text}")
    print(f"\nTespit edilen entity tipleri: {sorted({v.entity_type for v in result.violations})}")


if __name__ == "__main__":
    demo_programmatic_api()
    demo_yaml_api()

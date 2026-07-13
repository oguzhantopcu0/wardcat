"""
wardcat usage examples.

The sample text is intentionally Turkish — wardcat is Turkish-first (TC ID,
Turkish NER, Turkish phone/IBAN formats), so a Turkish sample shows entities
like TC_ID that only exist in that context.
"""

from wardcat import Action, Entity, Wardcat

SAMPLE = """
Merhaba, ben Ahmet Yılmaz. Şirketimizin sunucu IP'si 10.0.0.42.
Bana fatih.demir@example.com adresinden veya 0555 000 00 00 numarasından ulaşabilirsin.
Ödeme için IBAN: TR580000001111111111111111 kullanabilirsiniz.
Kredi kartım: 4111111111111111. TC kimliğim: 10987654321.
""".strip()


def demo_programmatic_api():
    print("=" * 60)
    print("Programmatic API")
    print("=" * 60)

    guard = (
        Wardcat(salt="example-salt-123")
        .add_entity(Entity.EMAIL, Action.WARN)
        .add_entity(Entity.CREDIT_CARD, Action.HASH)
        .add_entity(Entity.IBAN, Action.HASH)
        .add_entity(Entity.TC_ID, Action.HASH)
    )

    result = guard.scan(SAMPLE)

    print(f"\nSanitized text:\n{result.sanitized_text}")
    print(f"\nTotal violations: {len(result.violations)}")
    for v in result.violations:
        if v.action == "hash":
            print(f"  [{v.action}] {v.entity_type}: '{v.original}' → '{v.replacement}'")
        else:
            print(f"  [{v.action}] {v.entity_type}: '{v.original}'")


def demo_yaml_api():
    print("\n" + "=" * 60)
    print("Declarative (YAML) API")
    print("=" * 60)

    guard = Wardcat(config_path="config/default.yaml")
    result = guard.scan(SAMPLE)

    print(f"\nSanitized text:\n{result.sanitized_text}")
    print(f"\nDetected entity types: {sorted({v.entity_type for v in result.violations})}")


if __name__ == "__main__":
    demo_programmatic_api()
    demo_yaml_api()

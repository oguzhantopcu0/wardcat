from ai_guard import LLMGuard


def guard():
    return LLMGuard(use_ner=False)


def test_batch_returns_correct_count():
    texts = ["merhaba", "kart: 4111111111111111", "a@b.com"]
    results = guard().scan_batch(texts)
    assert len(results) == 3


def test_batch_clean_text():
    results = guard().scan_batch(["merhaba dünya"])
    assert results[0].is_clean


def test_batch_detects_independently():
    texts = [
        "email: a@b.com",
        "kart: 4111111111111111",
        "temiz metin",
    ]
    results = guard().scan_batch(texts)
    assert any(v.entity_type == "EMAIL" for v in results[0].violations)
    assert any(v.entity_type == "CREDIT_CARD" for v in results[1].violations)
    assert results[2].is_clean


def test_batch_empty_list():
    assert guard().scan_batch([]) == []


def test_batch_preserves_order():
    texts = [f"email{i}@test.com" for i in range(5)]
    results = guard().scan_batch(texts)
    for i, result in enumerate(results):
        assert result.violations[0].original == f"email{i}@test.com"

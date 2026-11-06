from wardcat import Entity, Wardcat


def guard():
    return Wardcat().add_entity(Entity.ALL, action="warn")


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


# ── An item that cannot be scanned must not come back as "sanitized" ─────────
# Its sanitized_text used to be the raw input, so a caller that did not check
# scan_error sent unscanned PII on to the model.


def _small_guard():
    from wardcat import Action

    guard = Wardcat(salt="s").add_entity(Entity.EMAIL, Action.REDACT)
    guard._config["max_text_bytes"] = 100
    guard._rebuild()
    return guard


OVERSIZED = "ali@example.com " * 20


def test_an_oversized_item_has_no_sanitized_text():
    results = _small_guard().scan_batch(["a@b.io", OVERSIZED, "c@d.io"])
    assert results[1].scan_error
    assert results[1].sanitized_text == ""
    assert results[1].original_text == OVERSIZED
    assert results[0].sanitized_text == "[EMAIL]"
    assert results[2].sanitized_text == "[EMAIL]"


def test_an_oversized_item_has_no_sanitized_text_async():
    import asyncio

    results = asyncio.run(_small_guard().scan_batch_async(["a@b.io", OVERSIZED, "c@d.io"]))
    assert results[1].scan_error
    assert results[1].sanitized_text == ""
    assert results[1].original_text == OVERSIZED
    assert results[0].sanitized_text == "[EMAIL]"
    assert results[2].sanitized_text == "[EMAIL]"


def _crashing_on(guard, needle: str):
    real_apply = guard._engine._anonymizer.apply

    def apply(text, spans, **kwargs):
        if needle in text:
            raise RuntimeError("anonymizer crashed")
        return real_apply(text, spans, **kwargs)

    guard._engine._anonymizer.apply = apply
    return guard


def test_an_item_that_crashes_has_no_sanitized_text():
    from wardcat import Action

    guard = _crashing_on(Wardcat(salt="s").add_entity(Entity.EMAIL, Action.REDACT), "boom")
    results = guard.scan_batch(["a@b.io", "boom ali@example.com"])
    assert results[1].scan_error == "RuntimeError: anonymizer crashed"
    assert results[1].sanitized_text == ""
    assert results[0].sanitized_text == "[EMAIL]"


def test_an_item_that_crashes_has_no_sanitized_text_async():
    import asyncio

    from wardcat import Action

    guard = _crashing_on(Wardcat(salt="s").add_entity(Entity.EMAIL, Action.REDACT), "boom")
    results = asyncio.run(guard.scan_batch_async(["a@b.io", "boom ali@example.com"]))
    assert results[1].scan_error
    assert results[1].sanitized_text == ""
    assert results[0].sanitized_text == "[EMAIL]"

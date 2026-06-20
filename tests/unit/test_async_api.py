"""
Async scan API tests.

Covers scan_async() and scan_batch_async() — both run in a thread pool
so they are safe to call from async code without blocking the event loop.
"""

from __future__ import annotations

import asyncio

import pytest

from ai_guard import AIGuard


@pytest.fixture
def guard():
    return AIGuard(use_ner=False)


class TestScanAsync:
    def test_scan_async_returns_result(self, guard):
        result = asyncio.run(guard.scan_async("email: test@example.com"))
        assert any(v.entity_type == "EMAIL" for v in result.violations)

    def test_scan_async_clean_text(self, guard):
        result = asyncio.run(guard.scan_async("clean text with no PII"))
        assert result.is_clean

    def test_scan_async_multiple_awaits(self, guard):
        async def _run():
            r1 = await guard.scan_async("a@b.com")
            r2 = await guard.scan_async("clean text")
            return r1, r2

        r1, r2 = asyncio.run(_run())
        assert not r1.is_clean
        assert r2.is_clean

    def test_scan_async_concurrent(self, guard):
        """Multiple concurrent scan_async calls should all succeed."""

        async def _run():
            tasks = [
                guard.scan_async("email: a@b.com"),
                guard.scan_async("card: 4111111111111111"),
                guard.scan_async("clean text"),
            ]
            return await asyncio.gather(*tasks)

        results = asyncio.run(_run())
        assert len(results) == 3
        assert not results[0].is_clean
        assert not results[1].is_clean
        assert results[2].is_clean


class TestScanBatchAsync:
    def test_scan_batch_async_returns_correct_count(self, guard):
        texts = ["a@b.com", "4111111111111111", "clean"]
        results = asyncio.run(guard.scan_batch_async(texts))
        assert len(results) == 3

    def test_scan_batch_async_empty(self, guard):
        results = asyncio.run(guard.scan_batch_async([]))
        assert results == []

    def test_scan_batch_async_preserves_order(self, guard):
        texts = [f"email{i}@test.com" for i in range(5)]
        results = asyncio.run(guard.scan_batch_async(texts))
        for i, result in enumerate(results):
            assert result.violations[0].original == f"email{i}@test.com"

    def test_scan_batch_async_detects_pii(self, guard):
        texts = ["user@example.com", "no pii here"]
        results = asyncio.run(guard.scan_batch_async(texts))
        assert any(v.entity_type == "EMAIL" for v in results[0].violations)
        assert results[1].is_clean


class TestScanAsyncActions:
    """Test that hash/redact/mask actions work correctly through the async path."""

    def test_hash_action_in_async(self):
        guard = AIGuard(use_ner=False, salt="s")
        guard.add_entity("EMAIL", action="hash")
        result = asyncio.run(guard.scan_async("a@b.com"))
        assert "a@b.com" not in result.sanitized_text
        assert "[EMAIL:" in result.sanitized_text

    def test_redact_action_in_async(self):
        guard = AIGuard(use_ner=False)
        guard.add_entity("EMAIL", action="redact")
        result = asyncio.run(guard.scan_async("a@b.com"))
        assert result.sanitized_text == "[EMAIL]"

    def test_mask_action_in_async(self):
        guard = AIGuard(use_ner=False)
        guard.add_entity("EMAIL", action="mask")
        result = asyncio.run(guard.scan_async("a@b.com"))
        replacement = result.violations[0].replacement
        assert replacement is not None
        assert "*" in replacement

    def test_allowlist_in_async(self):
        guard = AIGuard(use_ner=False)
        guard.add_entity("EMAIL", action="warn")
        guard.add_allowlist(["a@b.com"])
        result = asyncio.run(guard.scan_async("a@b.com"))
        assert result.is_clean

    def test_denylist_in_async(self):
        guard = AIGuard(use_ner=False)
        guard.add_entity("PERSON", action="warn")
        guard.add_denylist([{"value": "Jane Doe", "entity_type": "PERSON"}])
        result = asyncio.run(guard.scan_async("Hello Jane Doe"))
        assert any(v.entity_type == "PERSON" for v in result.violations)

    def test_scan_async_oversized_input_raises(self):
        import tempfile

        import pytest
        import yaml

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml.dump({"max_text_bytes": 10}, f)
            cfg_path = f.name

        guard = AIGuard(config_path=cfg_path, use_ner=False)
        with pytest.raises(ValueError, match="too large"):
            asyncio.run(guard.scan_async("x" * 11))

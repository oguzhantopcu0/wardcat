"""
Async scan API tests.

Covers scan_async() and scan_batch_async() — both run in a thread pool
so they are safe to call from async code without blocking the event loop.
"""
from __future__ import annotations

import asyncio

import pytest

from ai_guard import LLMGuard


@pytest.fixture
def guard():
    return LLMGuard(use_ner=False)


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

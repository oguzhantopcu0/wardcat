"""
Batch and async scanning — regex-only (no external services required).

- scan_batch: scan many texts in parallel via a thread pool
- scan_async / scan_batch_async: async API for use inside an event loop
"""

import asyncio

from ai_guard import LLMGuard

TEXTS = [
    "Email me at ali@example.com",
    "Card: 4111 1111 1111 1111",
    "Just a clean sentence with no PII.",
    "IBAN TR33 0006 1005 1978 6457 8413 26",
]


def batch_demo() -> None:
    guard = LLMGuard(use_ner=False, salt="example-salt")
    results = guard.scan_batch(TEXTS)
    print("== scan_batch ==")
    for text, r in zip(TEXTS, results, strict=True):
        flag = "clean" if r.is_clean else f"{len(r.violations)} violation(s)"
        print(f"  [{flag:16}] {text}")


async def async_demo() -> None:
    guard = LLMGuard(use_ner=False, salt="example-salt")
    results = await guard.scan_batch_async(TEXTS)
    print("\n== scan_batch_async ==")
    for text, r in zip(TEXTS, results, strict=True):
        types = sorted({v.entity_type for v in r.violations})
        print(f"  {types or 'clean':}  ← {text}")


if __name__ == "__main__":
    batch_demo()
    asyncio.run(async_demo())

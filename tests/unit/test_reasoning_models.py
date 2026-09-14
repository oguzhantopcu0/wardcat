"""Reasoning models: thinking is switched off, and any that leaks through is ignored.

Measured before the fix on qwen3:14b (Ollama 0.33, M1 16 GB): one short scan took
71–492 s with thinking and 10–12 s without, with the same detections. Thinking
blew through the default timeout, so the layer simply did not run.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wardcat.detectors.llm_detector import LLMDetector
from wardcat.llm.backends.ollama import OllamaBackend
from wardcat.llm.prompt import parse_sensitivity, strip_reasoning


def _response(body: dict) -> MagicMock:
    response = MagicMock()
    response.json.return_value = body
    response.raise_for_status = MagicMock()
    return response


def test_ollama_generate_asks_for_no_thinking():
    with patch("httpx.post", return_value=_response({"response": "[]"})) as post:
        OllamaBackend(model="qwen3:14b").complete("x")
    assert post.call_args[1]["json"]["think"] is False


def test_ollama_async_generate_asks_for_no_thinking():
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.post = AsyncMock(return_value=_response({"response": "[]"}))
    with patch("httpx.AsyncClient", return_value=client):
        asyncio.run(OllamaBackend(model="qwen3:14b").complete_async("x"))
    assert client.post.call_args[1]["json"]["think"] is False


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (
            '<think>maybe [1, 2]?</think>\n[{"type":"EMAIL","text":"a@b.co"}]',
            '\n[{"type":"EMAIL","text":"a@b.co"}]',
        ),
        ("<THINK>upper-case tags</THINK>false", "false"),
        ("<think>a</think>x<think>b</think>y", "xy"),
        ("<think>cut off before it closed", ""),
        ("the template opened the block</think>true", "true"),
        ('[{"type":"PERSON","text":"Ali"}]', '[{"type":"PERSON","text":"Ali"}]'),
    ],
)
def test_strip_reasoning(reply, expected):
    assert strip_reasoning(reply) == expected


def test_detector_ignores_brackets_inside_reasoning():
    detector = LLMDetector(MagicMock(), {"EMAIL"})
    raw = (
        "<think>The user wants a list like [type, text]. There is one email.</think>"
        '[{"type":"EMAIL","text":"ali@example.com"}]'
    )
    assert detector._parse_llm_response(raw) == [{"type": "EMAIL", "text": "ali@example.com"}]


def test_a_no_while_thinking_does_not_decide_sensitivity():
    """The verdict comes from the answer, not from a word in the reasoning."""
    assert parse_sensitivity("<think>No card here... wait, an IBAN.</think>true") is True
    assert parse_sensitivity("<think>yes, looks like a name? no.</think>false") is False

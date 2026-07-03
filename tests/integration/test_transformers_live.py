"""Live Transformers-backend test — exercises the REAL pipeline on a tiny model.

The unit tests mock the pipeline, so nothing ever ran a real model — which is
exactly why the ``dtype`` vs ``torch_dtype`` kwarg bug (real inference failing on
every call) was invisible. This loads a tiny instruct model and runs one real
generation, catching that whole class of pipeline/kwarg bug in seconds.

Marked ``slow`` and auto-skipped when torch/transformers (the ``transformers``
extra) or the model are unavailable — so a normal ``pytest`` run and the default
CI are unaffected.

Run explicitly:
    uv run --extra transformers pytest -m slow tests/integration/test_transformers_live.py
Pick the model:
    WARDCAT_TEST_TRANSFORMERS_MODEL=... uv run pytest -m slow ...
"""

from __future__ import annotations

import os

import pytest

# Real model download + CPU inference legitimately exceed the suite's default
# 30s per-test timeout, so lift it for this module.
pytestmark = [pytest.mark.slow, pytest.mark.timeout(600)]

# Tiny instruct model with a chat template (~270 MB). Ungated.
MODEL = os.environ.get("WARDCAT_TEST_TRANSFORMERS_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")


@pytest.fixture(scope="module")
def backend():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from wardcat.llm.backends.transformers_backend import TransformersBackend

    be = TransformersBackend(MODEL)
    try:
        be._get_pipeline()  # load now so a download/load failure skips, not errors
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Transformers model {MODEL!r} unavailable: {exc}")
    return be


def test_pipeline_builds_and_generates(backend):
    # The real pipeline(...) construction + generate — the exact path where the
    # dtype/torch_dtype kwarg bug lived. A mock would never have caught it.
    out = backend.complete_messages(
        [{"role": "user", "content": "Reply with the single word: OK"}],
        timeout=120,
    )
    assert isinstance(out, str)
    assert out.strip() != ""


def test_backend_reports_its_model(backend):
    assert backend.list_models() == [MODEL]

"""GLiNER2 detector tests.

The gliner2 package (and torch) is an optional extra, so the model is mocked
throughout — these tests never load a real model. They cover label selection,
GLiNER-label → ai-guard-entity mapping, threshold filtering, confidence
capping, and the AIGuard wiring (with_gliner + policy).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_guard.detectors.gliner_detector import _MAX_CONFIDENCE, GLiNERDetector


class _FakeGliner:
    """Stand-in for a loaded GLiNER2 model."""

    def __init__(self, response: dict) -> None:
        self._response = response
        self.calls: list[tuple[str, list[str]]] = []

    def extract_entities(self, text, labels, include_confidence, include_spans):
        self.calls.append((text, list(labels)))
        assert include_confidence is True
        assert include_spans is True
        return self._response


def _make(enabled: set[str], response: dict | None = None, **kwargs):
    fake = _FakeGliner(response or {"entities": {}})
    with patch("ai_guard.detectors.gliner_detector._load_gliner_model", return_value=fake):
        det = GLiNERDetector(enabled, **kwargs)
    return det, fake


# ── Label selection ───────────────────────────────────────────────────────────


class TestLabelSelection:
    def test_only_enabled_entity_labels_requested(self):
        det, _ = _make({"PERSON", "EMAIL"})
        assert "email" in det._labels
        assert "person" in det._labels
        # full_name/first_name/... all map to PERSON, so they are requested too
        assert "first_name" in det._labels
        # IBAN not enabled → its label is not requested
        assert "iban" not in det._labels

    def test_no_supported_entities_means_no_model_call(self):
        # ORG is not in the GLiNER label map → nothing to ask the model
        det, fake = _make({"ORG"})
        assert det._labels == []
        assert det.detect("Contact Apple Inc.") == []
        assert fake.calls == []  # model is never invoked


# ── Detection & mapping ───────────────────────────────────────────────────────


class TestDetection:
    def test_maps_labels_to_entity_types(self):
        response = {
            "entities": {
                "email": [{"text": "a@b.com", "confidence": 0.9, "start": 0, "end": 7}],
                "person": [{"text": "Ali Veli", "confidence": 0.95, "start": 11, "end": 19}],
            }
        }
        det, fake = _make({"EMAIL", "PERSON"}, response)
        spans = det.detect("a@b.com by Ali Veli")
        by_type = {s.entity_type: s for s in spans}
        assert set(by_type) == {"EMAIL", "PERSON"}
        assert by_type["EMAIL"].start == 0 and by_type["EMAIL"].end == 7
        assert by_type["PERSON"].text == "Ali Veli"
        assert fake.calls  # model was invoked

    def test_threshold_filters_low_confidence(self):
        response = {
            "entities": {
                "email": [
                    {"text": "a@b.com", "confidence": 0.9, "start": 0, "end": 7},
                    {"text": "c@d.com", "confidence": 0.3, "start": 11, "end": 18},
                ]
            }
        }
        det, _ = _make({"EMAIL"}, response, threshold=0.5)
        spans = det.detect("a@b.com or c@d.com")
        assert len(spans) == 1
        assert spans[0].text == "a@b.com"

    def test_confidence_capped_below_regex(self):
        response = {
            "entities": {"email": [{"text": "a@b.com", "confidence": 0.999, "start": 0, "end": 7}]}
        }
        det, _ = _make({"EMAIL"}, response)
        span = det.detect("a@b.com")[0]
        assert span.confidence == _MAX_CONFIDENCE  # never ties/beats a 1.0 regex span

    def test_span_without_offsets_is_skipped(self):
        response = {
            "entities": {
                "email": [
                    {"text": "a@b.com", "confidence": 0.9},  # no start/end
                    {"text": "c@d.com", "confidence": 0.9, "start": 11, "end": 18},
                ]
            }
        }
        det, _ = _make({"EMAIL"}, response)
        spans = det.detect("a@b.com or c@d.com")
        assert len(spans) == 1
        assert spans[0].text == "c@d.com"

    def test_unknown_label_in_response_is_ignored(self):
        response = {
            "entities": {
                "drivers_license_number": [  # mapped to nothing in ai-guard
                    {"text": "X123", "confidence": 0.9, "start": 0, "end": 4}
                ]
            }
        }
        det, _ = _make({"EMAIL"}, response)
        assert det.detect("X123") == []

    def test_empty_response_returns_empty(self):
        det, _ = _make({"EMAIL"}, {"entities": {}})
        assert det.detect("nothing here") == []


class _PosGliner:
    """Fake model that returns a PERSON span for every occurrence of a token,
    with chunk-relative offsets — used to verify chunking + re-basing."""

    def __init__(self, token: str) -> None:
        self.token = token
        self.calls: list[str] = []

    def extract_entities(self, text, labels, include_confidence, include_spans):
        self.calls.append(text)
        items = []
        i = text.find(self.token)
        while i != -1:
            items.append(
                {"text": self.token, "confidence": 0.95, "start": i, "end": i + len(self.token)}
            )
            i = text.find(self.token, i + 1)
        return {"entities": {"person": items}}


class TestChunking:
    def test_short_text_single_chunk(self):
        fake = _PosGliner("Zoe")
        with patch("ai_guard.detectors.gliner_detector._load_gliner_model", return_value=fake):
            det = GLiNERDetector({"PERSON"}, chunk_size=1000)
        det.detect("Hi Zoe, welcome.")
        assert len(fake.calls) == 1  # not split

    def test_long_text_is_chunked_with_global_offsets(self):
        fake = _PosGliner("Zoe")
        with patch("ai_guard.detectors.gliner_detector._load_gliner_model", return_value=fake):
            det = GLiNERDetector({"PERSON"}, chunk_size=50)
        # Two "Zoe"s far apart so they land in different windows.
        text = ("x" * 40) + " Zoe " + ("y" * 80) + " Zoe end"
        spans = det.detect(text)
        assert len(fake.calls) >= 2  # actually split into multiple windows
        # Every reported span maps back to the real global slice.
        for s in spans:
            assert text[s.start : s.end] == "Zoe"
        starts = {s.start for s in spans}
        assert text.index("Zoe") in starts
        assert text.index("Zoe", 60) in starts

    def test_chunk_size_zero_guard_does_not_hang(self):
        # size//2 == 0 overlap; ensure progress is still made (no infinite loop)
        fake = _PosGliner("Zoe")
        with patch("ai_guard.detectors.gliner_detector._load_gliner_model", return_value=fake):
            det = GLiNERDetector({"PERSON"}, chunk_size=10)
        spans = det.detect("Zoe " * 20)
        assert spans  # found some, and returned


# ── AIGuard wiring ────────────────────────────────────────────────────────────


class TestGuardWiring:
    def test_with_gliner_enables_layer_config(self):
        from ai_guard import AIGuard

        guard = AIGuard(use_ner=False)
        with patch("ai_guard.detectors.gliner_detector._load_gliner_model"):
            guard.with_gliner(threshold=0.4).add_entity("PERSON", "hash")
        cfg = guard._config["gliner_detector"]
        assert cfg["enabled"] is True
        assert cfg["threshold"] == 0.4

    def test_with_gliner_builds_detector_when_entity_enabled(self):
        from ai_guard import AIGuard
        from ai_guard.detectors.gliner_detector import GLiNERDetector as _GD

        with patch("ai_guard.detectors.gliner_detector._load_gliner_model"):
            guard = AIGuard(use_ner=False).with_gliner().add_entity("PERSON", "hash")
        assert any(isinstance(d, _GD) for d in guard._detectors)

    def test_gliner_layer_off_builds_no_detector(self):
        from ai_guard import AIGuard
        from ai_guard.detectors.gliner_detector import GLiNERDetector as _GD

        guard = AIGuard(use_ner=False).add_entity("PERSON", "hash")  # no with_gliner
        assert not any(isinstance(d, _GD) for d in guard._detectors)

    def test_load_failure_is_skipped_not_fatal(self):
        from ai_guard import AIGuard
        from ai_guard.detectors.gliner_detector import GLiNERDetector as _GD

        # Simulate the optional dependency being absent.
        with patch(
            "ai_guard.detectors.gliner_detector._load_gliner_model",
            side_effect=ImportError("No module named 'gliner2'"),
        ):
            guard = AIGuard(use_ner=False).with_gliner().add_entity("PERSON", "hash")
        # Guard still builds; the GLiNER detector is simply skipped.
        assert not any(isinstance(d, _GD) for d in guard._detectors)

    def test_end_to_end_scan_hashes_person(self):
        from ai_guard import AIGuard

        response = {
            "entities": {
                "person": [{"text": "Ali Veli", "confidence": 0.95, "start": 6, "end": 14}]
            }
        }
        fake = _FakeGliner(response)
        with patch("ai_guard.detectors.gliner_detector._load_gliner_model", return_value=fake):
            guard = AIGuard(salt="s", use_ner=False).with_gliner().add_entity("PERSON", "hash")
            result = guard.scan("Name: Ali Veli")
        assert "Ali Veli" not in result.sanitized_text
        assert "[PERSON:" in result.sanitized_text


# ── Discoverability / policy ──────────────────────────────────────────────────


class TestGlinerPolicy:
    def test_supported_entities_gliner_layer(self):
        from ai_guard import AIGuard
        from ai_guard.core.registry import GLINER_ENTITIES

        assert AIGuard.supported_entities("gliner") == GLINER_ENTITIES
        assert "PERSON" in AIGuard.supported_entities("gliner")
        assert "EMAIL" in AIGuard.supported_entities("gliner")

    def test_gliner_is_a_valid_layer(self):
        from ai_guard.core.registry import VALID_LAYERS

        assert "gliner" in VALID_LAYERS

    def test_add_entity_gliner_layer_enables_shared_flag(self):
        from ai_guard import AIGuard

        guard = AIGuard(use_ner=False)
        with patch("ai_guard.detectors.gliner_detector._load_gliner_model"):
            guard.with_gliner().add_entity("PERSON", "hash", layers=["gliner"])
        assert guard._config["entities"]["PERSON"]["enabled"] is True

    def test_invalid_threshold_rejected(self):
        from ai_guard import AIGuard
        from ai_guard.exceptions import ConfigError

        guard = AIGuard(use_ner=False)
        guard._config["gliner_detector"] = {"enabled": True, "model": "x", "threshold": 5}
        with pytest.raises(ConfigError, match="threshold"):
            from ai_guard.config.loader import validate_config

            validate_config(guard._config)

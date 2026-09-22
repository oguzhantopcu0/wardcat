"""with_strict(): a scan that covers less than configured is an error."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from wardcat import Action, DegradedScanError, Entity, Wardcat, WardcatError
from wardcat.llm.backends.base import BaseLLMBackend


def _dead_backend() -> BaseLLMBackend:
    backend = MagicMock(spec=BaseLLMBackend)
    backend.complete_messages.side_effect = ConnectionError("backend down")

    async def _down(*_a, **_k):
        raise ConnectionError("backend down")

    backend.complete_messages_async.side_effect = _down
    return backend


def _guard_with_dead_llm(*, strict: bool) -> Wardcat:
    guard = (
        Wardcat(salt="s")
        .with_llm(model="x", circuit_failures=0)
        .add_entity(Entity.EMAIL, Action.REDACT)
    )
    guard._engine.detectors[-1].backend = _dead_backend()
    if strict:
        guard.with_strict()
        guard._engine.detectors[-1].backend = _dead_backend()
    return guard


class TestBuildTime:
    def test_a_model_that_fails_to_load_refuses_the_guard(self, monkeypatch) -> None:
        import wardcat.guard as guard_module

        monkeypatch.setattr(
            guard_module, "_resolve_spacy_model", lambda m: (_ for _ in ()).throw(OSError("no"))
        )
        with pytest.raises(DegradedScanError) as info:
            (
                Wardcat(salt="s")
                .with_strict()
                .with_ner(spacy_model="en_core_web_sm", auto_download=False)
                .add_entity(Entity.PERSON, Action.REDACT)
            )
        assert info.value.result is None
        assert any("could not be loaded" in w for w in info.value.warnings)

    def test_it_is_a_wardcat_error(self) -> None:
        assert issubclass(DegradedScanError, WardcatError)

    def test_yaml_key(self, tmp_path) -> None:
        cfg = tmp_path / "policy.yaml"
        cfg.write_text("strict: true\nentities:\n  EMAIL: {enabled: true, action: redact}\n")
        guard = Wardcat(config_path=str(cfg), salt="s")
        assert guard._config["strict"] is True

    def test_yaml_rejects_a_non_boolean(self, tmp_path) -> None:
        from wardcat import ConfigError

        cfg = tmp_path / "policy.yaml"
        cfg.write_text("strict: yes please\n")
        with pytest.raises(ConfigError):
            Wardcat(config_path=str(cfg), salt="s")


class TestScanTime:
    def test_a_layer_that_fails_mid_scan_raises_with_the_partial_result(self) -> None:
        guard = _guard_with_dead_llm(strict=True)

        with pytest.raises(DegradedScanError) as info:
            guard.scan("mail ali@example.com")

        assert info.value.result is not None
        assert info.value.result.sanitized_text == "mail [EMAIL]"
        assert any("did not run" in w for w in info.value.warnings)

    def test_async_path_raises_too(self) -> None:
        guard = _guard_with_dead_llm(strict=True)

        with pytest.raises(DegradedScanError):
            asyncio.run(guard.scan_async("mail ali@example.com"))

    def test_scan_batch_does_not_swallow_it(self) -> None:
        guard = _guard_with_dead_llm(strict=True)

        with pytest.raises(DegradedScanError):
            guard.scan_batch(["a ali@example.com", "b"])

    def test_scan_batch_async_does_not_swallow_it(self) -> None:
        guard = _guard_with_dead_llm(strict=True)

        with pytest.raises(DegradedScanError):
            asyncio.run(guard.scan_batch_async(["a ali@example.com", "b"]))

    def test_without_strict_the_same_scan_returns_a_warning(self) -> None:
        guard = _guard_with_dead_llm(strict=False)

        result = guard.scan("mail ali@example.com")

        assert result.sanitized_text == "mail [EMAIL]"
        assert result.warnings

    def test_a_healthy_strict_guard_scans_normally(self) -> None:
        guard = Wardcat(salt="s").with_strict().add_entity(Entity.EMAIL, Action.REDACT)

        result = guard.scan("mail ali@example.com")

        assert result.sanitized_text == "mail [EMAIL]"
        assert result.warnings == []

    def test_strict_can_be_switched_off_again(self) -> None:
        guard = _guard_with_dead_llm(strict=True).with_strict(False)
        guard._engine.detectors[-1].backend = _dead_backend()

        assert guard.scan("mail ali@example.com").warnings

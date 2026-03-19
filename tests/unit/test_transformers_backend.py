"""
TransformersBackend birim testleri.

Tüm testler torch ve transformers kütüphanelerini içe aktarmadan
unittest.mock.patch ve MagicMock kullanarak backend'i test eder.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch, call

import pytest

from ai_guard.llm.backends.transformers_backend import TransformersBackend


# ── complete_messages() ──────────────────────────────────────────────────────

class TestCompleteMessages:
    def _make_backend(self) -> TransformersBackend:
        return TransformersBackend(model="meta-llama/Llama-3.2-1B-Instruct")

    def test_calls_pipeline_with_messages(self):
        backend = self._make_backend()
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"generated_text": "test response"}]

        with patch.object(backend, "_get_pipeline", return_value=mock_pipe):
            result = backend.complete_messages(
                [{"role": "user", "content": "hello"}]
            )

        mock_pipe.assert_called_once_with(
            [{"role": "user", "content": "hello"}],
            max_new_tokens=512,
            do_sample=False,
            return_full_text=False,
        )
        assert result == "test response"

    def test_returns_last_chat_message_content(self):
        """Chat format çıktısında son mesajın içeriği döndürülmeli."""
        backend = self._make_backend()
        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            {
                "generated_text": [
                    {"role": "user",      "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ]
            }
        ]

        with patch.object(backend, "_get_pipeline", return_value=mock_pipe):
            result = backend.complete_messages(
                [{"role": "user", "content": "hello"}]
            )

        assert result == "world"

    def test_custom_max_new_tokens(self):
        backend = TransformersBackend(
            model="meta-llama/Llama-3.2-1B-Instruct",
            max_new_tokens=128,
        )
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"generated_text": "ok"}]

        with patch.object(backend, "_get_pipeline", return_value=mock_pipe):
            backend.complete_messages([{"role": "user", "content": "hi"}])

        _, kwargs = mock_pipe.call_args
        assert kwargs["max_new_tokens"] == 128


# ── complete() ───────────────────────────────────────────────────────────────

class TestComplete:
    def test_delegates_to_complete_messages(self):
        """complete() should wrap the prompt in a user message and call complete_messages."""
        backend = TransformersBackend(model="meta-llama/Llama-3.2-1B-Instruct")

        with patch.object(backend, "complete_messages", return_value="answer") as mock_cm:
            result = backend.complete("my prompt", timeout=30)

        mock_cm.assert_called_once_with(
            [{"role": "user", "content": "my prompt"}],
            timeout=30,
        )
        assert result == "answer"


# ── list_models() ────────────────────────────────────────────────────────────

class TestListModels:
    def test_returns_model_name(self):
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        backend = TransformersBackend(model=model_id)
        assert backend.list_models() == [model_id]

    def test_returns_list_type(self):
        backend = TransformersBackend(model="some-model")
        result = backend.list_models()
        assert isinstance(result, list)
        assert len(result) == 1


# ── Lazy loading ─────────────────────────────────────────────────────────────

class TestLazyLoading:
    def test_pipeline_not_loaded_on_init(self):
        backend = TransformersBackend(model="meta-llama/Llama-3.2-1B-Instruct")
        assert backend._pipeline is None

    def test_pipeline_loaded_on_first_call(self):
        backend = TransformersBackend(model="meta-llama/Llama-3.2-1B-Instruct")
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"generated_text": "hi"}]

        with patch.object(backend, "_load_pipeline", return_value=mock_pipe) as mock_load:
            backend.complete_messages([{"role": "user", "content": "test"}])
            backend.complete_messages([{"role": "user", "content": "test2"}])

        # _load_pipeline sadece bir kez çağrılmalı
        mock_load.assert_called_once()

    def test_pipeline_reused_on_subsequent_calls(self):
        backend = TransformersBackend(model="meta-llama/Llama-3.2-1B-Instruct")
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"generated_text": "hi"}]

        with patch.object(backend, "_load_pipeline", return_value=mock_pipe):
            backend.complete_messages([{"role": "user", "content": "first"}])
            backend.complete_messages([{"role": "user", "content": "second"}])

        # Pipeline iki kez çağrılmış olmalı (iki mesaj)
        assert mock_pipe.call_count == 2


# ── pull_model() ─────────────────────────────────────────────────────────────

class TestPullModel:
    def test_calls_snapshot_download(self):
        backend = TransformersBackend(model="meta-llama/Llama-3.2-1B-Instruct")
        model_id = "meta-llama/Llama-3.1-8B-Instruct"

        with patch(
            "ai_guard.llm.backends.transformers_backend.snapshot_download",
            create=True,
        ) as mock_dl:
            # huggingface_hub modülünü mock'la
            mock_hf_hub = MagicMock()
            mock_hf_hub.snapshot_download = mock_dl
            with patch.dict(
                sys.modules,
                {"huggingface_hub": mock_hf_hub},
            ):
                backend.pull_model(model_id)

    def test_pull_model_calls_snapshot_download_with_repo_id(self):
        backend = TransformersBackend(model="meta-llama/Llama-3.2-1B-Instruct")
        model_id = "meta-llama/Llama-3.1-8B-Instruct"

        mock_snapshot = MagicMock()
        mock_hf_hub = MagicMock()
        mock_hf_hub.snapshot_download = mock_snapshot

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf_hub}):
            backend.pull_model(model_id)

        mock_snapshot.assert_called_once_with(repo_id=model_id)

    def test_pull_model_raises_import_error_without_huggingface_hub(self):
        backend = TransformersBackend(model="meta-llama/Llama-3.2-1B-Instruct")

        # huggingface_hub modülünü import edilemez hale getir
        original = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="huggingface_hub"):
                backend.pull_model("any-model")
        finally:
            if original is None:
                sys.modules.pop("huggingface_hub", None)
            else:
                sys.modules["huggingface_hub"] = original

    def test_pull_model_calls_on_progress_callback(self):
        backend = TransformersBackend(model="meta-llama/Llama-3.2-1B-Instruct")
        progress_calls = []

        def on_progress(p):
            progress_calls.append(p.status)

        mock_snapshot = MagicMock()
        mock_hf_hub = MagicMock()
        mock_hf_hub.snapshot_download = mock_snapshot

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf_hub}):
            backend.pull_model("some-model", on_progress=on_progress)

        assert "downloading" in progress_calls
        assert "success"     in progress_calls


# ── ImportError when transformers not installed ──────────────────────────────

class TestImportError:
    def test_import_error_raised_when_transformers_missing(self):
        """transformers kurulu değilse _load_pipeline ImportError fırlatmalı."""
        backend = TransformersBackend(model="meta-llama/Llama-3.2-1B-Instruct")

        # transformers ve torch'u import edilemez hale getir
        with patch.dict(sys.modules, {"transformers": None, "torch": None}):  # type: ignore[dict-item]
            with pytest.raises(ImportError, match="transformers"):
                backend._load_pipeline()


# ── Edge cases: empty / malformed pipeline output ────────────────────────────

class TestOutputEdgeCases:
    def _make_backend(self) -> TransformersBackend:
        return TransformersBackend(model="meta-llama/Llama-3.2-1B-Instruct")

    def test_raises_on_empty_outputs_list(self):
        """Pipeline boş liste döndürürse ValueError fırlatılmalı."""
        backend = self._make_backend()
        mock_pipe = MagicMock(return_value=[])

        with patch.object(backend, "_get_pipeline", return_value=mock_pipe):
            with pytest.raises(ValueError, match="boş çıktı"):
                backend.complete_messages([{"role": "user", "content": "hi"}])

    def test_raises_on_empty_chat_format_list(self):
        """Chat format: generated_text boş listeyse ValueError fırlatılmalı."""
        backend = self._make_backend()
        mock_pipe = MagicMock(return_value=[{"generated_text": []}])

        with patch.object(backend, "_get_pipeline", return_value=mock_pipe):
            with pytest.raises(ValueError, match="boş"):
                backend.complete_messages([{"role": "user", "content": "hi"}])

    def test_missing_content_key_returns_empty_string(self):
        """Chat format: son mesajda 'content' yoksa boş string döner."""
        backend = self._make_backend()
        mock_pipe = MagicMock(return_value=[
            {"generated_text": [{"role": "assistant"}]}  # content yok
        ])

        with patch.object(backend, "_get_pipeline", return_value=mock_pipe):
            result = backend.complete_messages([{"role": "user", "content": "hi"}])

        assert result == ""

    def test_raises_on_pipeline_exception(self):
        """Pipeline RuntimeError fırlatırsa ValueError'a çevrilmeli."""
        backend = self._make_backend()
        mock_pipe = MagicMock(side_effect=RuntimeError("CUDA out of memory"))

        with patch.object(backend, "_get_pipeline", return_value=mock_pipe):
            with pytest.raises(ValueError, match="inference başarısız"):
                backend.complete_messages([{"role": "user", "content": "hi"}])

    def test_generated_text_as_plain_string(self):
        """Plain string çıktı (non-chat model) string olarak döner."""
        backend = self._make_backend()
        mock_pipe = MagicMock(return_value=[{"generated_text": "plain output"}])

        with patch.object(backend, "_get_pipeline", return_value=mock_pipe):
            result = backend.complete_messages([{"role": "user", "content": "hi"}])

        assert result == "plain output"


# ── Chat template validation ──────────────────────────────────────────────────

class TestChatTemplateValidation:
    def test_warns_when_no_chat_template(self, caplog):
        """Tokenizer'da chat_template yoksa uyarı loglanmalı."""
        import logging
        backend = TransformersBackend(model="meta-llama/Llama-3.2-1B-Instruct")

        mock_tokenizer = MagicMock()
        del mock_tokenizer.chat_template  # attribute yok

        mock_pipe = MagicMock()
        mock_pipe.tokenizer = mock_tokenizer

        with caplog.at_level(logging.WARNING, logger="ai_guard.llm.backends.transformers_backend"):
            with patch.object(backend, "_load_pipeline", return_value=mock_pipe):
                backend._get_pipeline()

        assert any("chat_template" in r.message for r in caplog.records)

    def test_no_warning_when_chat_template_present(self, caplog):
        """Tokenizer'da chat_template varsa uyarı loglanmamalı."""
        import logging
        backend = TransformersBackend(model="meta-llama/Llama-3.2-1B-Instruct")

        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = "{% for message in messages %}..."

        mock_pipe = MagicMock()
        mock_pipe.tokenizer = mock_tokenizer

        with caplog.at_level(logging.WARNING, logger="ai_guard.llm.backends.transformers_backend"):
            with patch.object(backend, "_load_pipeline", return_value=mock_pipe):
                backend._get_pipeline()

        chat_template_warnings = [r for r in caplog.records if "chat_template" in r.message]
        assert len(chat_template_warnings) == 0

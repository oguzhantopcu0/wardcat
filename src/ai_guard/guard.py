from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_guard.config.loader import load_config
from ai_guard.core.engine import DetectionEngine
from ai_guard.core.models import ScanResult, warn_unknown_entity
from ai_guard.detectors.base import BaseDetector
from ai_guard.detectors.regex_detector import RegexDetector

logger = logging.getLogger(__name__)


def _resolve_spacy_model(model: str) -> str:
    """İstenen SpaCy modeli kurulu değilse alternatif önerir.

    Davranış:
    - Model kuruluysa olduğu gibi döndürür.
    - Kurulu değilse mevcut SpaCy modellerini listeler ve uyarı verir.
    - Hiç model bulunamazsa orijinal adı döndürür (NERDetector kendi hatasını üretir).
    """
    try:
        import spacy
        spacy.load(model)
        return model
    except OSError:
        pass

    # Kurulu modelleri tara
    try:
        import spacy.util
        installed = list(spacy.util.get_installed_models())
    except Exception:
        installed = []

    if not installed:
        logger.warning(
            "SpaCy modeli bulunamadı: %r. Kurulum: python -m spacy download %s",
            model, model,
        )
        return model

    # Dil ön ekine göre eşleştir (tr_, en_, vb.)
    lang_prefix = model.split("_")[0] + "_"
    same_lang = [m for m in installed if m.startswith(lang_prefix)]
    fallback = same_lang[0] if same_lang else installed[0]

    logger.warning(
        "SpaCy modeli %r kurulu değil. Alternatif kullanılıyor: %r. "
        "Doğru model için: python -m spacy download %s",
        model, fallback, model,
    )
    return fallback


# Hangi entity'nin hangi dedektöre ait olduğunu merkezi tablo
_REGEX_ENTITIES = {
    "CREDIT_CARD", "EMAIL", "PHONE", "IBAN", "IP_ADDRESS", "IPv6",
    "TC_ID", "ADDRESS", "POSTAL_CODE",
    "UUID", "SSN", "MAC_ADDRESS", "JWT", "NIN", "CUSTOM_SECRET",
}
_NER_ENTITIES   = {"PERSON", "ORG", "ADDRESS"}


class LLMGuard:
    """
    Kullanıcıya sunulan ana arayüz.

    Programmatic API (method zinciri)::

        guard = (
            LLMGuard(salt=os.environ["LLMGUARD_SALT"])
            .configure_entity("EMAIL",       enabled=True,  action="hash")
            .configure_entity("CREDIT_CARD", enabled=True,  action="hash")
            .configure_entity("ORG",         enabled=False)
        )
        result = guard.scan(text)

    Declarative API (YAML)::

        guard = LLMGuard(config_path="config/my_policy.yaml")
        result = guard.scan(text)

    Ortam değişkenleri (YAML ve constructor argümanlarının üzerine yazar)::

        LLMGUARD_SALT          — hash tuzlama değeri
        LLMGUARD_LLM_URL       — Ollama/OpenAI-compat servis URL'i
        LLMGUARD_LLM_MODEL     — LLM model adı
        LLMGUARD_LLM_API_KEY   — API anahtarı (OpenAI-compat)
        LLMGUARD_LLM_TIMEOUT   — LLM timeout (saniye, varsayılan 60)
        LLMGUARD_SPACY_MODEL   — SpaCy model adı

    LLM dedektörü (Ollama)::

        guard = LLMGuard(
            use_llm=True,
            llm_model="llama3.1:8b",
            llm_base_url="http://localhost:11434",
        )
        result = guard.scan(text)
    """

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        salt: str = "",
        use_ner: bool = True,
        spacy_model: str = "en_core_web_sm",
        use_llm: bool = False,
        llm_backend: str = "ollama",               # "ollama" | "openai_compatible" | "transformers"
        llm_model: str = "llama3.2",
        llm_base_url: str = "http://localhost:11434",
        llm_api_key: str = "",
        llm_timeout: int = 60,
        auto_pull: bool = False,          # Ollama: model yoksa otomatik indir
        llm_device_map: str = "auto",     # Transformers: GPU dağılımı
        llm_load_in_8bit: bool = False,   # Transformers: 8-bit quantization
        llm_load_in_4bit: bool = False,   # Transformers: 4-bit quantization
    ) -> None:
        self._config = load_config(config_path)

        # Constructor argümanları YAML'ı override eder
        # (ortam değişkenleri load_config içinde zaten uygulandı)
        if salt:
            self._config["salt"] = salt
        if not use_ner:
            self._config["use_ner"] = False
        if spacy_model != "en_core_web_sm":
            self._config["spacy_model"] = spacy_model

        # LLM dedektörü override'ları
        llm_cfg = self._config.setdefault("llm_detector", {})
        if use_llm:
            llm_cfg["enabled"] = True
        if llm_backend != "ollama":
            llm_cfg["backend"] = llm_backend
        if llm_model != "llama3.2":
            llm_cfg["model"] = llm_model
        if llm_base_url != "http://localhost:11434":
            llm_cfg["base_url"] = llm_base_url
        if llm_api_key:
            llm_cfg["api_key"] = llm_api_key
        if llm_timeout != 60:
            llm_cfg["timeout"] = llm_timeout
        if auto_pull:
            llm_cfg["auto_pull"] = True
        if llm_device_map != "auto":
            llm_cfg["device_map"] = llm_device_map
        if llm_load_in_8bit:
            llm_cfg["load_in_8bit"] = True
        if llm_load_in_4bit:
            llm_cfg["load_in_4bit"] = True

        # Salt uyarısı: hash aksiyonu var ama salt boşsa
        effective_salt = self._config.get("salt", "")
        if not effective_salt:
            entity_cfg = self._config.get("entities", {})
            has_hash = any(
                cfg.get("action") == "hash"
                for cfg in entity_cfg.values()
                if isinstance(cfg, dict)
            )
            if has_hash:
                logger.warning(
                    "Hash salt boş — aynı PII değerleri her zaman aynı hash'i üretir. "
                    "Production'da LLMGUARD_SALT ortam değişkenini ayarlayın."
                )

        self._rebuild()

    # ------------------------------------------------------------------
    # Tarama
    # ------------------------------------------------------------------

    def scan(self, text: str) -> ScanResult:
        """Metni tara ve ScanResult döndür."""
        return self._engine.scan(text)

    def scan_batch(self, texts: List[str]) -> List[ScanResult]:
        """
        Birden fazla metni sırayla tara.

        Her metin bağımsız olarak taranır; tek bir öğedeki hata
        diğerlerini etkilemez — hata olan öğe için orijinal metin
        dokunulmadan döndürülür.

        :param texts: Taranacak metin listesi
        :returns:     Her metne karşılık gelen ``ScanResult`` listesi
        """
        results: List[ScanResult] = []
        for i, text in enumerate(texts):
            try:
                results.append(self._engine.scan(text))
            except Exception as exc:
                logger.error(
                    "scan_batch öğe %d başarısız oldu, orijinal metin döndürülüyor: %s",
                    i, exc, exc_info=True,
                )
                results.append(ScanResult(
                    original_text=text,
                    sanitized_text=text,
                    violations=[],
                ))
        return results

    # ------------------------------------------------------------------
    # Programmatic API
    # ------------------------------------------------------------------

    def configure_entity(
        self,
        entity_type: str,
        enabled: bool = True,
        action: str = "warn",
    ) -> "LLMGuard":
        """
        Tek bir entity tipini yapılandır. Method chaining destekler.

        :param entity_type: Örn. "EMAIL", "PERSON", "CREDIT_CARD"
        :param enabled:     Bu entity'yi tarama motoruna dahil et
        :param action:      "warn" veya "hash"
        """
        warn_unknown_entity(entity_type)
        if action not in ("warn", "hash"):
            raise ValueError(
                f"Geçersiz action {action!r}. Geçerli değerler: 'warn', 'hash'"
            )
        self._config.setdefault("entities", {})[entity_type] = {
            "enabled": enabled,
            "action": action,
        }
        self._rebuild()
        return self

    def set_salt(self, salt: str) -> "LLMGuard":
        """Hash tuzunu güncelle."""
        self._config["salt"] = salt
        self._rebuild()
        return self

    # ------------------------------------------------------------------
    # Dahili
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        """Konfigürasyon değiştiğinde dedektörleri ve engine'i yeniden kur."""
        self._detectors: List[BaseDetector] = []
        entity_cfg = self._config.get("entities", {})

        # Regex dedektörü
        enabled_regex = {
            e for e in _REGEX_ENTITIES
            if entity_cfg.get(e, {}).get("enabled", True)
        }
        if enabled_regex:
            self._detectors.append(RegexDetector(enabled_regex))

        # SpaCy NER dedektörü (opsiyonel)
        if self._config.get("use_ner", True):
            enabled_ner = {
                e for e in _NER_ENTITIES
                if entity_cfg.get(e, {}).get("enabled", True)
            }
            if enabled_ner:
                try:
                    from ai_guard.detectors.ner_detector import NERDetector
                    model = self._config.get("spacy_model", "en_core_web_sm")
                    model = _resolve_spacy_model(model)
                    self._detectors.append(NERDetector(enabled_ner, model))
                except Exception as exc:
                    logger.warning(
                        "SpaCy NER yüklenemedi, yalnızca regex kullanılıyor. Hata: %s", exc
                    )

        # LLM dedektörü (opsiyonel)
        llm_cfg = self._config.get("llm_detector", {})
        if llm_cfg.get("enabled", False):
            self._detectors.append(self._build_llm_detector(llm_cfg))

        self._engine = DetectionEngine(self._config, self._detectors)

    def _build_llm_detector(self, llm_cfg: Dict[str, Any]) -> BaseDetector:
        """LLM dedektörünü konfigürasyona göre kur."""
        from ai_guard.detectors.llm_detector import LLMDetector

        backend_name = llm_cfg.get("backend", "ollama")
        model        = llm_cfg.get("model",    "llama3.2")
        base_url     = llm_cfg.get("base_url", "http://localhost:11434")
        api_key      = llm_cfg.get("api_key",  "")
        timeout      = llm_cfg.get("timeout",  60)

        if backend_name == "ollama":
            from ai_guard.llm.backends.ollama import OllamaBackend
            from ai_guard.llm.model_manager import ModelManager
            ollama_backend = OllamaBackend(base_url=base_url, model=model)
            if llm_cfg.get("auto_pull", False):
                mgr = ModelManager(ollama_backend)
                mgr.ensure_available(model, verbose=True)
            backend = ollama_backend
        elif backend_name == "openai_compatible":
            from ai_guard.llm.backends.openai_compat import OpenAICompatBackend
            backend = OpenAICompatBackend(base_url=base_url, model=model, api_key=api_key)
        elif backend_name == "transformers":
            from ai_guard.llm.backends.transformers_backend import TransformersBackend
            backend = TransformersBackend(
                model=model,
                device_map=llm_cfg.get("device_map", "auto"),
                load_in_8bit=llm_cfg.get("load_in_8bit", False),
                load_in_4bit=llm_cfg.get("load_in_4bit", False),
            )
        else:
            raise ValueError(
                f"Bilinmeyen LLM backend: {backend_name!r}. "
                "Geçerli değerler: 'ollama', 'openai_compatible', 'transformers'"
            )

        entity_cfg = llm_cfg.get("entities", {})
        enabled = {
            e for e, cfg in entity_cfg.items()
            if cfg.get("enabled", True)
        }

        # LLM entity aksiyonlarını global engine config'e ekle (override yapmadan)
        for entity, cfg in entity_cfg.items():
            self._config["entities"].setdefault(entity, cfg)

        return LLMDetector(backend=backend, enabled_entities=enabled, timeout=timeout)

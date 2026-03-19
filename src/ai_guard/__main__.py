"""
CLI giriş noktası.

Kullanım:
    python -m ai_guard scan --text "Merhaba, kartım 4111111111111111"
    python -m ai_guard scan --file girdi.txt --config config/policy.yaml
    python -m ai_guard scan --text "..." --salt "gizli" --no-ner --format json
    python -m ai_guard scan --text "..." --llm --llm-model llama3.1:8b
    python -m ai_guard scan --text "..." --llm --llm-model llama3.1:8b --auto-pull
    python -m ai_guard batch --file satirlar.txt --format json
    python -m ai_guard models list
    python -m ai_guard models list --recommended
    python -m ai_guard models setup
    python -m ai_guard models pull llama3.1:8b
    python -m ai_guard models pull llama3.1:8b --llm-url http://10.0.0.5:11434

Ortam değişkenleri:
    LLMGUARD_SALT         — --salt yerine
    LLMGUARD_LLM_URL      — --llm-url yerine
    LLMGUARD_LLM_MODEL    — --llm-model yerine
    LLMGUARD_LLM_API_KEY  — --llm-api-key yerine
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai_guard",
        description="LLM girdilerindeki hassas verileri tara ve anonimleştir.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── ortak argümanlar ────────────────────────────────────────────────
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config",  metavar="PATH", help="YAML politika dosyası")
    common.add_argument("--salt",    default="",     help="Hash tuzlama değeri (LLMGUARD_SALT env ile de verilebilir)")
    common.add_argument("--no-ner",  action="store_true", help="SpaCy NER'ı devre dışı bırak")
    common.add_argument("--model",   default="en_core_web_sm", metavar="MODEL",
                        help="SpaCy modeli (varsayılan: en_core_web_sm)")
    common.add_argument("--format",  choices=["text", "json"], default="text",
                        help="Çıktı formatı (varsayılan: text)")
    common.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Log seviyesi (varsayılan: WARNING)")
    # LLM dedektörü ortak argümanları
    common.add_argument("--llm",         action="store_true",
                        help="LLM dedektörünü etkinleştir (Ollama veya OpenAI-compat)")
    common.add_argument("--llm-backend", default="ollama",
                        choices=["ollama", "openai_compatible", "transformers"],
                        metavar="BACKEND", help="LLM backend (varsayılan: ollama)")
    common.add_argument("--llm-model",   default="llama3.1:8b", metavar="MODEL",
                        help="LLM model adı (varsayılan: llama3.1:8b)")
    common.add_argument("--llm-url",     default="", metavar="URL",
                        help="LLM servis URL'i (varsayılan: LLMGUARD_LLM_URL veya http://localhost:11434)")
    common.add_argument("--llm-api-key", default="", metavar="KEY",
                        help="OpenAI-compat API anahtarı (LLMGUARD_LLM_API_KEY env ile de verilebilir)")
    common.add_argument("--auto-pull",   action="store_true",
                        help="Model mevcut değilse Ollama'dan otomatik indir")

    # ── scan ────────────────────────────────────────────────────────────
    p_scan = sub.add_parser("scan", parents=[common], help="Tek bir metni tara")
    src = p_scan.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", metavar="TEXT",  help="Doğrudan metin girdisi")
    src.add_argument("--file", metavar="PATH",  type=Path, help="Metin dosyası")

    # ── batch ───────────────────────────────────────────────────────────
    p_batch = sub.add_parser("batch", parents=[common],
                              help="Dosyadaki her satırı ayrı metin olarak tara")
    p_batch.add_argument("--file", metavar="PATH", type=Path, required=True,
                         help="Her satırı bağımsız metin olan dosya")

    # ── models ──────────────────────────────────────────────────────────
    p_models = sub.add_parser("models", help="On-prem LLM model yönetimi")
    models_sub = p_models.add_subparsers(dest="models_command", required=True)

    # models list
    p_list = models_sub.add_parser("list", help="Mevcut modelleri listele")
    p_list.add_argument("--llm-backend", default="ollama",
                        choices=["ollama", "openai_compatible", "transformers"], metavar="BACKEND")
    p_list.add_argument("--llm-url", default="", metavar="URL")
    p_list.add_argument("--llm-api-key", default="", metavar="KEY")
    p_list.add_argument("--recommended", action="store_true",
                        help="Önerilen modelleri katalogdan listele (Ollama bağlantısı gerekmez)")

    # models setup
    p_setup = models_sub.add_parser(
        "setup",
        help="Önerilen modeller arasından seçim yap ve indir",
    )
    p_setup.add_argument("--llm-url", default="", metavar="URL",
                         help="Ollama servis URL'i")
    p_setup.add_argument("--non-interactive", action="store_true",
                         help="Onay sormadan varsayılan modeli indir")

    # models pull
    p_pull = models_sub.add_parser("pull", help="Model indir (Ollama)")
    p_pull.add_argument("model_name", help="İndirilecek model, örn. llama3.1:8b")
    p_pull.add_argument("--llm-url", default="", metavar="URL",
                        help="Ollama servis URL'i")

    return parser


def _resolve_llm_url(args_url: str) -> str:
    """--llm-url → LLMGUARD_LLM_URL env → varsayılan."""
    if args_url:
        return args_url
    return os.environ.get("LLMGUARD_LLM_URL", "http://localhost:11434")


def _make_guard(args: argparse.Namespace):
    from ai_guard import LLMGuard
    return LLMGuard(
        config_path=args.config,
        salt=args.salt,
        use_ner=not args.no_ner,
        spacy_model=args.model,
        use_llm=args.llm,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        llm_base_url=_resolve_llm_url(args.llm_url),
        llm_api_key=args.llm_api_key or os.environ.get("LLMGUARD_LLM_API_KEY", ""),
        auto_pull=getattr(args, "auto_pull", False),
    )


def _make_backend(args: argparse.Namespace):
    backend = getattr(args, "llm_backend", "ollama")
    url     = _resolve_llm_url(getattr(args, "llm_url", ""))
    api_key = getattr(args, "llm_api_key", "") or os.environ.get("LLMGUARD_LLM_API_KEY", "")

    if backend == "ollama":
        from ai_guard.llm.backends.ollama import OllamaBackend
        return OllamaBackend(base_url=url)
    else:
        from ai_guard.llm.backends.openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(base_url=url, model="", api_key=api_key)


# ── çıktı formatlayıcılar ───────────────────────────────────────────────

def _result_to_dict(result) -> dict:
    return {
        "is_clean": result.is_clean,
        "sanitized_text": result.sanitized_text,
        "violations": [
            {
                "entity_type": v.entity_type,
                "original":    v.original,
                "start":       v.start,
                "end":         v.end,
                "action":      v.action.value,
                "replacement": v.replacement,
            }
            for v in result.violations
        ],
    }


def _print_text(result, label: str | None = None) -> None:
    if label:
        print(f"\n── {label} ──")
    print(f"Temizlenmiş metin:\n  {result.sanitized_text}")
    if result.violations:
        print(f"İhlaller ({len(result.violations)}):")
        for v in result.violations:
            if v.replacement:
                print(f"  [{v.action.value:4}] {v.entity_type}: '{v.original}' → '{v.replacement}'")
            else:
                print(f"  [{v.action.value:4}] {v.entity_type}: '{v.original}'")
    else:
        print("  Hassas veri bulunamadı.")


# ── komut işleyiciler ───────────────────────────────────────────────────

def _read_file(path: Path) -> str:
    """Dosyayı oku; encoding hatası veya dosya bulunamazsa anlaşılır hata ver."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Dosya UTF-8 olarak okunamadı: {path}\n"
            f"Detay: {exc}\n"
            "İpucu: Dosyayı UTF-8 olarak kaydedin veya encoding'i dönüştürün."
        )


def cmd_scan(args: argparse.Namespace) -> None:
    """Handle the ``scan`` sub-command — scan a single text for PII."""
    guard = _make_guard(args)
    text  = args.text if args.text else _read_file(args.file)
    result = guard.scan(text)

    if args.format == "json":
        print(json.dumps(_result_to_dict(result), ensure_ascii=False, indent=2))
    else:
        _print_text(result)


def cmd_batch(args: argparse.Namespace) -> None:
    """Handle the ``batch`` sub-command — scan each line of a file independently."""
    guard = _make_guard(args)
    content = _read_file(args.file)
    lines = [ln for ln in content.splitlines() if ln.strip()]
    results = guard.scan_batch(lines)

    if args.format == "json":
        output = [
            {"line": i + 1, "text": line, **_result_to_dict(r)}
            for i, (line, r) in enumerate(zip(lines, results))
        ]
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        for i, (line, result) in enumerate(zip(lines, results), 1):
            _print_text(result, label=f"Satır {i}: {line[:60]}{'…' if len(line) > 60 else ''}")


def cmd_models(args: argparse.Namespace) -> None:
    """Handle the ``models`` sub-command — list, setup, or pull on-prem LLM models."""
    from ai_guard.llm.model_manager import ModelManager

    if args.models_command == "list":
        if getattr(args, "recommended", False):
            _print_catalog()
            return
        backend = _make_backend(args)
        mgr     = ModelManager(backend)
        models  = mgr.list()
        if models:
            print("Ollama'da mevcut modeller:")
            for m in models:
                print(f"  • {m}")
        else:
            print("Hiç model bulunamadı.")
        print("\nÖnerilen modeller için: python -m ai_guard models list --recommended")

    elif args.models_command == "setup":
        _cmd_setup(args)

    elif args.models_command == "pull":
        from ai_guard.llm.backends.ollama import OllamaBackend
        url     = _resolve_llm_url(args.llm_url)
        backend = OllamaBackend(base_url=url)
        mgr     = ModelManager(backend)
        mgr.pull(args.model_name, verbose=True)


def _print_catalog() -> None:
    """Önerilen model listesini terminale yaz."""
    from ai_guard.llm.model_catalog import CATALOG
    print("Önerilen on-prem LLM modelleri:\n")
    print(f"  {'#':<3} {'Model':<20} {'VRAM':<8} Açıklama")
    print(f"  {'-'*3} {'-'*20} {'-'*8} {'-'*40}")
    for i, m in enumerate(CATALOG, 1):
        tag  = " ← önerilen" if m.recommended else ""
        vram = f"~{m.vram_gb:.1f} GB"
        print(f"  {i:<3} {m.name:<20} {vram:<8} {m.description}{tag}")
    print()


def _cmd_setup(args: argparse.Namespace) -> None:
    """İnteraktif model seçim ve indirme akışı."""
    from ai_guard.llm.model_catalog import CATALOG, recommended
    from ai_guard.llm.model_manager import ModelManager
    from ai_guard.llm.backends.ollama import OllamaBackend

    _print_catalog()

    non_interactive = getattr(args, "non_interactive", False)

    if non_interactive:
        chosen = recommended()
        print(f"Varsayılan model seçildi: {chosen.name}")
    else:
        try:
            raw = input(
                f"Model numarası seçin [1-{len(CATALOG)}, "
                f"varsayılan=1 ({CATALOG[0].name})]: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nİptal edildi.")
            return

        if raw == "":
            chosen = CATALOG[0]
        else:
            try:
                idx = int(raw) - 1
                if not (0 <= idx < len(CATALOG)):
                    raise ValueError
                chosen = CATALOG[idx]
            except ValueError:
                print(f"Geçersiz seçim: '{raw}'. Kurulum iptal edildi.")
                return

    print(f"\nSeçilen model: {chosen.name} (~{chosen.vram_gb:.1f} GB VRAM)\n")

    url     = _resolve_llm_url(args.llm_url)
    backend = OllamaBackend(base_url=url)
    mgr     = ModelManager(backend)

    ok = mgr.ensure_available(chosen.name, verbose=True)
    if ok:
        print(f"\nKullanım:\n"
              f"  python -m ai_guard scan --text \"...\" "
              f"--llm --llm-model {chosen.name}\n"
              f"\nPython API:\n"
              f"  guard = LLMGuard(use_llm=True, llm_model=\"{chosen.name}\")")


# ── main ────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point — parse arguments and dispatch to the appropriate command handler."""
    parser = _build_parser()
    args = parser.parse_args()

    # Logging konfigürasyonu — CLI başlatınca aktif olur
    log_level = getattr(args, "log_level", "WARNING")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        if args.command == "scan":
            cmd_scan(args)
        elif args.command == "batch":
            cmd_batch(args)
        elif args.command == "models":
            cmd_models(args)
    except FileNotFoundError as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        sys.exit(1)
    except ConnectionError as exc:
        print(f"Bağlantı hatası: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nİptal edildi.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        logger.exception("Beklenmeyen hata")
        print(f"Beklenmeyen hata: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

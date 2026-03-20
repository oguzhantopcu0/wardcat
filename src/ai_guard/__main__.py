"""
CLI entry point.

Usage:
    python -m ai_guard scan --text "Hello, my card is 4111111111111111"
    python -m ai_guard scan --file input.txt --config config/policy.yaml
    python -m ai_guard scan --text "..." --salt "secret" --no-ner --format json
    python -m ai_guard scan --text "..." --llm --llm-model llama3.1:8b
    python -m ai_guard scan --text "..." --llm --llm-model llama3.1:8b --auto-pull
    python -m ai_guard batch --file lines.txt --format json
    python -m ai_guard models list
    python -m ai_guard models list --recommended
    python -m ai_guard models setup
    python -m ai_guard models pull llama3.1:8b
    python -m ai_guard models pull llama3.1:8b --llm-url http://10.0.0.5:11434

Environment variables:
    LLMGUARD_SALT         — instead of --salt
    LLMGUARD_LLM_URL      — instead of --llm-url
    LLMGUARD_LLM_MODEL    — instead of --llm-model
    LLMGUARD_LLM_API_KEY  — instead of --llm-api-key
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
        description="Scan and anonymize sensitive data in LLM inputs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── common arguments ────────────────────────────────────────────────
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config",  metavar="PATH", help="YAML policy file")
    common.add_argument("--salt",    default="",     help="Hash salt value (can also be set via LLMGUARD_SALT env)")
    common.add_argument("--no-ner",  action="store_true", help="Disable SpaCy NER")
    common.add_argument("--model",   default="en_core_web_sm", metavar="MODEL",
                        help="SpaCy model (default: en_core_web_sm)")
    common.add_argument("--format",  choices=["text", "json"], default="text",
                        help="Output format (default: text)")
    common.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Log level (default: WARNING)")
    # LLM detector common arguments
    common.add_argument("--llm",         action="store_true",
                        help="Enable LLM detector (Ollama or OpenAI-compat)")
    common.add_argument("--llm-backend", default="ollama",
                        choices=["ollama", "openai_compatible", "transformers"],
                        metavar="BACKEND", help="LLM backend (default: ollama)")
    common.add_argument("--llm-model",   default="llama3.1:8b", metavar="MODEL",
                        help="LLM model name (default: llama3.1:8b)")
    common.add_argument("--llm-url",     default="", metavar="URL",
                        help="LLM service URL (default: LLMGUARD_LLM_URL or http://localhost:11434)")
    common.add_argument("--llm-api-key", default="", metavar="KEY",
                        help="OpenAI-compat API key (can also be set via LLMGUARD_LLM_API_KEY env)")
    common.add_argument("--auto-pull",   action="store_true",
                        help="Automatically download from Ollama if model is not available")

    # ── scan ────────────────────────────────────────────────────────────
    p_scan = sub.add_parser("scan", parents=[common], help="Scan a single text")
    src = p_scan.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", metavar="TEXT",  help="Direct text input")
    src.add_argument("--file", metavar="PATH",  type=Path, help="Text file")

    # ── batch ───────────────────────────────────────────────────────────
    p_batch = sub.add_parser("batch", parents=[common],
                              help="Scan each line in a file as a separate text")
    p_batch.add_argument("--file", metavar="PATH", type=Path, required=True,
                         help="File where each line is an independent text")

    # ── models ──────────────────────────────────────────────────────────
    p_models = sub.add_parser("models", help="On-prem LLM model management")
    models_sub = p_models.add_subparsers(dest="models_command", required=True)

    # models list
    p_list = models_sub.add_parser("list", help="List available models")
    p_list.add_argument("--llm-backend", default="ollama",
                        choices=["ollama", "openai_compatible", "transformers"], metavar="BACKEND")
    p_list.add_argument("--llm-url", default="", metavar="URL")
    p_list.add_argument("--llm-api-key", default="", metavar="KEY")
    p_list.add_argument("--recommended", action="store_true",
                        help="List recommended models from catalog (no Ollama connection required)")

    # models setup
    p_setup = models_sub.add_parser(
        "setup",
        help="Select from recommended models and download",
    )
    p_setup.add_argument("--llm-url", default="", metavar="URL",
                         help="Ollama service URL")
    p_setup.add_argument("--non-interactive", action="store_true",
                         help="Download default model without prompting for confirmation")

    # models pull
    p_pull = models_sub.add_parser("pull", help="Download a model (Ollama)")
    p_pull.add_argument("model_name", help="Model to download, e.g. llama3.1:8b")
    p_pull.add_argument("--llm-url", default="", metavar="URL",
                        help="Ollama service URL")

    return parser


def _resolve_llm_url(args_url: str) -> str:
    """--llm-url → LLMGUARD_LLM_URL env → default."""
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


# ── output formatters ───────────────────────────────────────────────────

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
    print(f"Sanitized text:\n  {result.sanitized_text}")
    if result.violations:
        print(f"Violations ({len(result.violations)}):")
        for v in result.violations:
            if v.replacement:
                print(f"  [{v.action.value:4}] {v.entity_type}: '{v.original}' → '{v.replacement}'")
            else:
                print(f"  [{v.action.value:4}] {v.entity_type}: '{v.original}'")
    else:
        print("  No sensitive data found.")


# ── command handlers ────────────────────────────────────────────────────

def _read_file(path: Path) -> str:
    """Read file; give a clear error if encoding fails or file is not found."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"File could not be read as UTF-8: {path}\n"
            f"Detail: {exc}\n"
            "Hint: Save the file as UTF-8 or convert its encoding."
        )


def _warn_if_no_salt(args: argparse.Namespace) -> None:
    """Write a warning to stderr if no salt is used in production."""
    salt = args.salt or os.environ.get("LLMGUARD_SALT", "")
    if not salt:
        print(
            "WARNING: Hash salt is not set — identical PII values will always produce the same "
            "hash. Use --salt or LLMGUARD_SALT in production.",
            file=sys.stderr,
        )


def cmd_scan(args: argparse.Namespace) -> None:
    """Handle the ``scan`` sub-command — scan a single text for PII."""
    _warn_if_no_salt(args)
    guard = _make_guard(args)
    text  = args.text if args.text else _read_file(args.file)
    result = guard.scan(text)

    if args.format == "json":
        print(json.dumps(_result_to_dict(result), ensure_ascii=False, indent=2))
    else:
        _print_text(result)


def cmd_batch(args: argparse.Namespace) -> None:
    """Handle the ``batch`` sub-command — scan each line of a file independently."""
    _warn_if_no_salt(args)
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
            _print_text(result, label=f"Line {i}: {line[:60]}{'…' if len(line) > 60 else ''}")


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
            print("Models available in Ollama:")
            for m in models:
                print(f"  • {m}")
        else:
            print("No models found.")
        print("\nFor recommended models: python -m ai_guard models list --recommended")

    elif args.models_command == "setup":
        _cmd_setup(args)

    elif args.models_command == "pull":
        from ai_guard.llm.backends.ollama import OllamaBackend
        url     = _resolve_llm_url(args.llm_url)
        backend = OllamaBackend(base_url=url)
        mgr     = ModelManager(backend)
        mgr.pull(args.model_name, verbose=True)


def _print_catalog() -> None:
    """Print the recommended model list to the terminal."""
    from ai_guard.llm.model_catalog import CATALOG
    print("Recommended on-prem LLM models:\n")
    print(f"  {'#':<3} {'Model':<20} {'VRAM':<8} Description")
    print(f"  {'-'*3} {'-'*20} {'-'*8} {'-'*40}")
    for i, m in enumerate(CATALOG, 1):
        tag  = " ← recommended" if m.recommended else ""
        vram = f"~{m.vram_gb:.1f} GB"
        print(f"  {i:<3} {m.name:<20} {vram:<8} {m.description}{tag}")
    print()


def _cmd_setup(args: argparse.Namespace) -> None:
    """Interactive model selection and download flow."""
    from ai_guard.llm.model_catalog import CATALOG, recommended
    from ai_guard.llm.model_manager import ModelManager
    from ai_guard.llm.backends.ollama import OllamaBackend

    _print_catalog()

    non_interactive = getattr(args, "non_interactive", False)

    if non_interactive:
        chosen = recommended()
        print(f"Default model selected: {chosen.name}")
    else:
        try:
            raw = input(
                f"Select model number [1-{len(CATALOG)}, "
                f"default=1 ({CATALOG[0].name})]: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
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
                print(f"Invalid selection: '{raw}'. Setup cancelled.")
                return

    print(f"\nSelected model: {chosen.name} (~{chosen.vram_gb:.1f} GB VRAM)\n")

    url     = _resolve_llm_url(args.llm_url)
    backend = OllamaBackend(base_url=url)
    mgr     = ModelManager(backend)

    ok = mgr.ensure_available(chosen.name, verbose=True)
    if ok:
        print(f"\nUsage:\n"
              f"  python -m ai_guard scan --text \"...\" "
              f"--llm --llm-model {chosen.name}\n"
              f"\nPython API:\n"
              f"  guard = LLMGuard(use_llm=True, llm_model=\"{chosen.name}\")")


# ── main ────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point — parse arguments and dispatch to the appropriate command handler."""
    parser = _build_parser()
    args = parser.parse_args()

    # Logging configuration — activated when CLI starts
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
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ConnectionError as exc:
        print(f"Connection error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        logger.exception("Unexpected error")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

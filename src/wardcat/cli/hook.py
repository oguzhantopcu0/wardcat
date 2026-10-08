"""``wardcat hook claude-code``: a Claude Code hook that keeps PII out of the session.

Register it for the events below; it reads the event as JSON on standard input
and answers in Claude Code's hook format.

``UserPromptSubmit``
    A prompt with a finding is blocked before it reaches the model. Claude Code
    cannot rewrite a prompt, so blocking is the only safe answer; the reason
    names the entity types, never the values.
``PreToolUse``
    A tool call whose arguments carry a finding (a Bash command, a file a Write
    would create, a URL) is denied, or put to the user with ``--tool-decision ask``.
``PostToolUse``
    The tool has already run, so its output is replaced with the sanitized text
    before the model reads it, using each entity type's action.

Register it only for these three events; any other is reported as a hook
error (exit 1) and left alone. The hook fails closed, per event: if the input cannot be read or the text
cannot be scanned, a prompt or tool call is blocked (exit 2) and a tool's
output is withheld. Claude Code itself lets an event through when a hook times
out, so keep the hook fast — the regex layer — or raise its ``timeout``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from wardcat.cli._guard_args import add_guard_args, build_guard
from wardcat.exceptions import ConfigError

if TYPE_CHECKING:  # pragma: no cover
    from wardcat import Wardcat

EXIT_BLOCK = 2  # Claude Code's "block this" exit code


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "hook",
        help="run as a coding-agent hook (claude-code) that keeps PII out of the session",
        allow_abbrev=False,
    )
    p.add_argument("agent", choices=["claude-code"], help="the agent calling the hook")
    add_guard_args(p, default_action="redact")
    p.add_argument(
        "--tool-decision",
        choices=["deny", "ask"],
        default="deny",
        help="for a tool call with a finding: deny it (default) or ask the user",
    )
    p.set_defaults(run=run)


def run(args: argparse.Namespace) -> int:
    from wardcat.cli import EXIT_CLEAN, EXIT_FOUND

    try:
        event = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        name = event["hook_event_name"]
    except (AttributeError, KeyError, TypeError, ValueError):
        return _block("could not read the hook input")
    handler = _HANDLERS.get(name)
    if handler is None:
        # Not a block: exit 2 means something else on other events ("keep going"
        # on Stop). Exit 1 is a visible hook error that changes nothing.
        print(f"wardcat: {name!r} is not an event this hook handles", file=sys.stderr)
        return EXIT_FOUND
    try:
        guard = build_guard(args, warn=False)
        out = handler(guard, event, args)
    except Exception as exc:  # fail closed: whatever went wrong, nothing passes unscanned
        reason = str(exc) if isinstance(exc, ConfigError) else type(exc).__name__
        if name == "PostToolUse":
            out = _replace_output(event, f"[wardcat withheld this output: {reason}]")
        else:
            return _block(f"could not scan ({reason})")
    if out is not None:
        print(json.dumps(out, ensure_ascii=False))
    return EXIT_CLEAN


def _block(reason: str) -> int:
    print(f"wardcat: {reason}; blocked", file=sys.stderr)
    return EXIT_BLOCK


def _on_prompt(guard: Wardcat, event: dict[str, Any], args: argparse.Namespace) -> Any:
    text = event.get("prompt_text", event.get("prompt"))
    if not isinstance(text, str):
        raise ConfigError("the event carries no prompt text")
    _, found = sanitize(guard, text)
    if not found:
        return None
    return {
        "decision": "block",
        "reason": f"wardcat: the prompt contains {_names(found)}; remove it and send again",
    }


def _on_tool_call(guard: Wardcat, event: dict[str, Any], args: argparse.Namespace) -> Any:
    tool_input = event.get("tool_input")
    if not isinstance(tool_input, dict):
        raise ConfigError("the event carries no tool input")
    found: set[str] = set()
    for text in _strings(tool_input):
        found |= sanitize(guard, text)[1]
    if not found:
        return None
    tool = event.get("tool_name", "tool")
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": args.tool_decision,
            "permissionDecisionReason": f"wardcat: the {tool} call contains {_names(found)}",
        }
    }


def _on_tool_output(guard: Wardcat, event: dict[str, Any], args: argparse.Namespace) -> Any:
    response = event.get("tool_response")
    if isinstance(response, dict) and response.get("type") == "image":
        return None  # nothing wardcat can read
    if isinstance(response, dict) and isinstance(response.get("text"), str):
        clean, found = sanitize(guard, response["text"])
    elif isinstance(response, str):
        clean, found = sanitize(guard, response)
    else:
        found = set()
        cleaned = _map_strings(response, lambda s: _collect(guard, s, found))
        clean = json.dumps(cleaned, ensure_ascii=False)
    if not found:
        return None
    out = _replace_output(event, clean)
    out["hookSpecificOutput"]["additionalContext"] = (
        f"wardcat replaced {_names(found)} in this tool output with placeholders"
    )
    return out


_HANDLERS: dict[str, Callable[[Wardcat, dict[str, Any], argparse.Namespace], Any]] = {
    "UserPromptSubmit": _on_prompt,
    "PreToolUse": _on_tool_call,
    "PostToolUse": _on_tool_output,
}


def _replace_output(event: dict[str, Any], text: str) -> dict[str, Any]:
    tool = str(event.get("tool_name", ""))
    key = "updatedMCPToolOutput" if tool.startswith("mcp__") else "updatedToolOutput"
    return {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            key: {"type": "text", "text": text},
        }
    }


def sanitize(guard: Wardcat, text: str) -> tuple[str, set[str]]:
    """The sanitized text and the entity types found, scanned in line-aligned pieces."""
    from wardcat.cli.check import _CHUNK_MARGIN, _pieces

    parts: list[str] = []
    found: set[str] = set()
    for _, piece in _pieces(text, guard.max_text_bytes - _CHUNK_MARGIN, "the text"):
        result = guard.scan(piece)
        if result.scan_error:
            raise ConfigError("the text could not be scanned")
        parts.append(result.sanitized_text)
        found.update(v.entity_type for v in result.violations)
    return "".join(parts), found


def _collect(guard: Wardcat, text: str, found: set[str]) -> str:
    clean, types = sanitize(guard, text)
    found |= types
    return clean


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _map_strings(value: Any, fn: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, dict):
        return {k: _map_strings(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [_map_strings(v, fn) for v in value]
    return value


def _names(found: set[str]) -> str:
    return ", ".join(sorted(found))

"""``wardcat`` alone, in a terminal: the interactive screen.

A prompt for trying a policy by hand: switch layers and filters on and off,
scan text, load a preset, and serve the result over HTTP while you work. Type
``/`` for a palette of commands that filters as you type; a status line under
the prompt shows the session, layers, filters and server. ``wardcat --resume
ID`` and ``wardcat --continue`` pick a saved session up again.

What it keeps out of files: scanned text never reaches the prompt history
(a ``scan`` line is not recorded), the salt is read from the environment
variable ``--salt-env`` names, and an LLM API key is read from one or typed
into a hidden prompt and kept in memory. See ``_sessions`` for what is saved.

Outside a terminal, ``wardcat`` with no command prints its usage and exits 2,
so a script that forgot its command fails instead of waiting for input.
"""

from __future__ import annotations

import argparse
import difflib
import os
import shlex
import sys
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NoReturn

from wardcat.cli import _sessions as sessions
from wardcat.cli._shell_ui import banner, candidates, help_text, paint, table
from wardcat.exceptions import ConfigError, WardcatError

if TYPE_CHECKING:  # pragma: no cover
    from wardcat import Wardcat

DEFAULT_FILTERS = (
    "CREDIT_CARD",
    "CUSTOM_SECRET",
    "EMAIL",
    "IBAN",
    "JWT",
    "PHONE",
    "TC_ID",
    "VEHICLE_PLATE",
)
_SALTED = {"hash", "tokenize", "surrogate"}
_BACKENDS = {
    "ollama": ("11434", False),
    "vllm": ("8000", True),
    "openai_compatible": ("8000", True),
}
_LOOPBACK = {"localhost", "127.0.0.1", "::1"}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The top-level options that open the screen (``wardcat --resume ID``)."""
    screen = parser.add_argument_group("interactive screen (wardcat with no command)")
    which = screen.add_mutually_exclusive_group()
    which.add_argument("--resume", dest="screen_resume", metavar="ID", help="resume a session")
    which.add_argument(
        "--continue",
        dest="screen_continue",
        action="store_true",
        help="resume the most recent session",
    )
    screen.add_argument(
        "--salt-env",
        dest="screen_salt_env",
        metavar="VAR",
        help="read the salt for hash, tokenize and surrogate from VAR",
    )


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from wardcat.cli import EXIT_CLEAN, EXIT_CONFIG

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        parser.print_usage(sys.stderr)
        print(
            "wardcat: error: give a command; the interactive screen needs a terminal",
            file=sys.stderr,
        )
        return EXIT_CONFIG
    shell = Shell.start(args, interactive=True)
    shell.loop()
    return EXIT_CLEAN


class _Cancelled(Exception):
    """The user left a question with Ctrl-C or Ctrl-D."""


class _Help(Exception):
    """A command's -h was answered; nothing else to do."""


class _Parser(argparse.ArgumentParser):
    """An argument parser for one screen command; it never exits the screen."""

    def error(self, message: str) -> NoReturn:
        raise ConfigError(f"{message}\n{self.format_usage().strip()}")

    def exit(self, status: int = 0, message: str | None = None) -> NoReturn:
        if message:
            print(message, end="")
        raise _Help


def _parser(prog: str, *arguments: tuple[tuple[str, ...], dict[str, Any]]) -> _Parser:
    p = _Parser(prog=prog, allow_abbrev=False)
    for names, options in arguments:
        p.add_argument(*names, **options)
    return p


class Shell:
    def __init__(self, state: dict[str, Any], *, salt: str = "", interactive: bool = False):
        import warnings

        # SpaCy's W094 (a model's loose version pin) is noise on a screen; a
        # model loads when the guard is built or first scans, so it is
        # silenced for the screen's whole life rather than around one call.
        warnings.filterwarnings("ignore", message=r"\[W094\]")
        self.state = state
        self.salt = salt
        self.api_key = ""  # typed into the screen; memory only
        self.interactive = interactive
        self.guard: Wardcat = self._make_guard()
        self._server: Any = None
        self._server_thread: threading.Thread | None = None
        self.port: int | None = None

    # ---------------------------------------------------------------- start

    @classmethod
    def start(cls, args: argparse.Namespace, *, interactive: bool) -> Shell:
        from wardcat import __version__

        notes: list[str] = []
        if args.screen_resume or args.screen_continue:
            if args.screen_continue:
                saved = sessions.all_sessions()
                if not saved:
                    raise ConfigError("no saved sessions yet; run `wardcat` to start one")
                state, had_salt = sessions.load(saved[0]["id"])
            else:
                state, had_salt = sessions.load(args.screen_resume)
            if had_salt:
                notes.append(
                    "this session was saved by wardcat-cli with its salt in the file; the salt "
                    "is not reused and the file is rewritten without it. Use --salt-env VAR."
                )
            opening = f"Resuming session {state['id']} (last used {state['updated_at']})"
        else:
            state = sessions.new_session()
            state["filters"] = dict.fromkeys(DEFAULT_FILTERS, "redact")
            opening = f"New session {state['id']}"
        salt = ""
        if args.screen_salt_env:
            state["salt_env"] = args.screen_salt_env
        if state["salt_env"]:
            salt = os.environ.get(state["salt_env"], "")
            if not salt:
                notes.append(f"{state['salt_env']} is not set; hashing runs without a salt")
        llm = state["llm"]
        if llm and llm.get("key_typed") and "llm" in state["layers"]:
            state["layers"].remove("llm")
            notes.append("the LLM API key typed last time is not saved; `add layer llm` again")
        if llm and llm.get("api_key_env") and not os.environ.get(llm["api_key_env"]):
            if "llm" in state["layers"]:
                state["layers"].remove("llm")
            notes.append(f"{llm['api_key_env']} is not set; the LLM layer is off")
        if interactive:
            print(banner(__version__))
            print()
        print(opening)
        for note in notes:
            print(paint(f"note: {note}", "yellow"))
        try:
            shell = cls(state, salt=salt, interactive=interactive)
        except ConfigError as exc:
            # A saved NER model or LLM setup that no longer works: open with
            # the regex layer rather than not at all.
            print(paint(f"note: {exc}; opening with the regex layer only", "yellow"))
            state["layers"] = ["regex"]
            state["ner"] = state["llm"] = None
            shell = cls(state, salt=salt, interactive=interactive)
        sessions.save(state)
        print(paint(shell.status() + " · type / for the menu, help for every command", "dim"))
        return shell

    # ---------------------------------------------------------------- guard

    def _make_guard(self, state: dict[str, Any] | None = None) -> Wardcat:
        from wardcat import Wardcat

        state = state or self.state
        guard = Wardcat(salt=self.salt) if self.salt else Wardcat()
        if state["ner"]:
            ner = state["ner"]
            sized = {"spacy_size": ner["size"]} if ner.get("size") else {}
            guard.with_ner(language=ner["language"], auto_download=False, **sized)
        if state["llm"]:
            guard.with_llm(**self._llm_kwargs(state["llm"]))
        if state["filters"]:
            guard.add_entities(dict(state["filters"]), layers=list(state["layers"]))
        return guard

    def _llm_kwargs(self, llm: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "backend": llm["backend"],
            "model": llm["model"],
            "allow_http": llm["allow_http"],
            "adjudicate": llm["adjudicate"],
        }
        if llm.get("base_url"):
            kwargs["base_url"] = llm["base_url"]
        if llm.get("api_key_env"):
            kwargs["api_key"] = os.environ.get(llm["api_key_env"], "")
        elif self.api_key:
            kwargs["api_key"] = self.api_key
        return kwargs

    def _apply(self, change: Callable[[dict[str, Any]], None]) -> None:
        """Change a copy of the state, build its guard, and keep both only if that works."""
        import copy

        candidate = copy.deepcopy(self.state)
        change(candidate)
        guard = self._make_guard(candidate)  # raises before anything is replaced
        self.state, self.guard = candidate, guard
        sessions.save(self.state)
        for warning in self._build_warnings():
            print(paint(f"warning: {warning}", "yellow"))
        if self._server is not None:
            self._restart_server()

    def _build_warnings(self) -> list[str]:
        engine = getattr(self.guard, "_engine", None)
        return list(getattr(engine, "build_warnings", ()) or ())

    def status(self) -> str:
        serving = f" · serving 127.0.0.1:{self.port}" if self.port else ""
        return (
            f"session {self.state['id']} · layers: {', '.join(self.state['layers'])} · "
            f"{len(self.state['filters'])} filters{serving}"
        )

    # ---------------------------------------------------------------- input

    def ask(self, question: str, *, default: str = "", secret: bool = False) -> str:
        """An answer; Ctrl-C or Ctrl-D cancels the command that asked."""
        try:
            return self._ask(question, default=default, secret=secret)
        except (EOFError, KeyboardInterrupt):
            print()
            raise _Cancelled from None

    def _ask(self, question: str, *, default: str, secret: bool) -> str:
        shown = f"{question} [{default}]: " if default and not secret else f"{question}: "
        if self.interactive:
            from prompt_toolkit import prompt

            answer = prompt(shown, is_password=secret)
        elif secret:
            import getpass

            answer = getpass.getpass(shown) if sys.stdin.isatty() else input(shown)
        else:
            answer = input(shown)
        return answer.strip() or default

    def loop(self, read_line: Callable[[], str] | None = None) -> None:
        read_line = read_line or self._reader()
        try:
            while True:
                try:
                    line = read_line().strip()
                except KeyboardInterrupt:
                    continue  # Ctrl-C clears the line; Ctrl-D or quit leaves
                except EOFError:
                    break
                if line.startswith("/"):
                    line = line[1:].strip()
                if not line:
                    continue
                command, _, rest = line.partition(" ")
                command = command.lower()
                if command in ("quit", "exit"):
                    break
                self.dispatch(command, rest.strip())
        finally:
            self._stop_server()
            sessions.save(self.state)
            print(
                f"\nsession saved: {self.state['id']} · resume with `wardcat --resume {self.state['id']}`"
            )

    def dispatch(self, command: str, rest: str) -> None:
        handler = _COMMANDS.get(command)
        if handler is None:
            close = difflib.get_close_matches(command, list(_COMMANDS), n=1)
            hint = f"; did you mean {close[0]}?" if close else "; type help"
            print(paint(f"unknown command {command!r}{hint}", "red"))
            return
        try:
            handler(self, rest)
        except _Help:
            pass
        except _Cancelled:
            print(paint("cancelled", "yellow"))
        except KeyboardInterrupt:
            print(paint("interrupted", "yellow"))
        except (ConfigError, WardcatError, ValueError, OSError) as exc:
            print(paint(f"error: {exc}", "red"))
        except Exception as exc:  # an unreachable backend, say: the screen stays up
            # Only the type: a message from below may quote what was scanned.
            print(paint(f"error: {type(exc).__name__}", "red"))

    def _reader(self) -> Callable[[], str]:
        if not self.interactive:
            return lambda: input(f"\nwardcat [{', '.join(self.state['layers'])}]> ")
        session = _prompt_session(self)
        from prompt_toolkit.formatted_text import ANSI

        return lambda: session.prompt(
            ANSI("\n" + paint("wardcat", "bold", "cyan") + " ❯ "),
            bottom_toolbar=lambda: f" {self.status()} · / menu · quit ",
        )

    # ---------------------------------------------------------------- scan

    def cmd_scan(self, rest: str) -> None:
        from wardcat.cli.check import _CHUNK_MARGIN, _pieces

        text = rest or self.ask("text to scan")
        if not text:
            return
        if not self.state["filters"]:
            print(paint("no filters are on; add one with `add filter ENTITY`", "yellow"))
        out, rows, warnings = [], [], {}
        for _, piece in _pieces(text, self.guard.max_text_bytes - _CHUNK_MARGIN, "the text"):
            result = self.guard.scan(piece)
            if result.scan_error:
                raise ConfigError("the text could not be scanned; nothing is shown")
            out.append(_highlight(result))
            warnings.update(dict.fromkeys(result.warnings))
            for v in result.violations:
                action = str(getattr(v.action, "value", v.action))
                source = str(getattr(v.source, "value", v.source))
                rows.append(
                    [
                        v.entity_type,
                        action,
                        v.replacement or "(kept)",
                        f"{v.confidence:.2f}",
                        source,
                    ]
                )
        print("".join(out))
        if rows:
            print()
            print(table(["Entity", "Action", "Replacement", "Confidence", "Layer"], rows))
        else:
            print(paint("nothing found", "green"))
        for warning in warnings:
            print(paint(f"warning: {warning}", "yellow"))

    # ---------------------------------------------------------------- filters

    def cmd_add(self, rest: str) -> None:
        what, _, rest = rest.partition(" ")
        if what == "filter":
            self._add_filter(rest)
        elif what == "layer":
            self._add_layer(rest)
        else:
            raise ConfigError("usage: add filter ENTITY [--action A] | add layer LAYER")

    def cmd_remove(self, rest: str) -> None:
        what, _, rest = rest.partition(" ")
        if what == "filter":
            self._remove_filter(rest)
        elif what == "layer":
            self._remove_layer(rest)
        else:
            raise ConfigError("usage: remove filter ENTITY | remove layer LAYER")

    def _add_filter(self, rest: str) -> None:
        from wardcat import Wardcat
        from wardcat.cli._guard_args import parse_entity_specs

        ns = _parser(
            "add filter",
            (("entity",), {"nargs": "+", "help": "EMAIL, or EMAIL,IBAN=mask"}),
            (("--action",), {"default": "redact", "help": "redact (default), mask, hash, ..."}),
        ).parse_args(shlex.split(rest))
        specs = [
            (name.upper(), action.lower())
            for name, action in parse_entity_specs(ns.entity, ns.action)
        ]
        from wardcat.core.actions import registered_actions

        actions = sorted(registered_actions())
        for _, action in specs:
            if action not in actions:
                raise ConfigError(f"unknown action {action!r}; choose from {', '.join(actions)}")
        known = Wardcat.supported_entities()
        for name, _ in specs:
            if name not in known:
                close = difflib.get_close_matches(name, sorted(known), n=1)
                hint = f"; did you mean {close[0]}?" if close else "; see `list filters`"
                raise ConfigError(f"unknown entity type {name!r}{hint}")

        def change(state: dict[str, Any]) -> None:
            state["filters"].update(specs)

        self._apply(change)
        for name, action in specs:
            print(paint(f"filter {name} on ({action})", "green"))
        if not self.salt and any(action in _SALTED for _, action in specs):
            print(paint("note: no salt; start with --salt-env VAR to salt hashes", "yellow"))

    def _remove_filter(self, rest: str) -> None:
        names = [n.strip().upper() for n in rest.replace(",", " ").split() if n.strip()]
        if not names:
            raise ConfigError("usage: remove filter ENTITY")
        missing = [n for n in names if n not in self.state["filters"]]
        if missing:
            raise ConfigError(f"not an active filter: {', '.join(missing)}")

        def change(state: dict[str, Any]) -> None:
            for n in names:
                del state["filters"][n]

        self._apply(change)
        print(paint(f"filter {', '.join(names)} off", "yellow"))

    def cmd_preset(self, rest: str) -> None:
        from wardcat.presets import PRESETS, get_preset

        if not rest:
            print(table(["Preset", "Covers"], [[n, get_preset(n).covers] for n in sorted(PRESETS)]))
            return
        preset = get_preset(rest.strip())

        def change(state: dict[str, Any]) -> None:
            state["filters"] = dict(preset.entities)

        self._apply(change)
        print(paint(f"preset {preset.name}: {len(preset.entities)} filters", "green"))
        missing = [
            x for x in sessions.LAYERS if x in preset.needs_layers and x not in self.state["layers"]
        ]
        if missing:
            needed = " and ".join(missing) + (" layers" if len(missing) > 1 else " layer")
            print(paint(f"note: some of its types are only found with the {needed}", "yellow"))
        if not self.salt and _SALTED & set(preset.entities.values()):
            print(paint("note: no salt; start with --salt-env VAR to salt hashes", "yellow"))

    def cmd_filters(self, rest: str) -> None:
        from wardcat import Wardcat

        if not self.state["filters"]:
            print("no filters are on")
            return
        support = {layer: Wardcat.supported_entities(layer) for layer in self.state["layers"]}
        rows = []
        for name, action in sorted(self.state["filters"].items()):
            found_by = [layer for layer, names in support.items() if name in names]
            rows.append([name, action, ", ".join(found_by) or "no active layer finds it"])
        print(table(["Entity", "Action", "Found by"], rows))

    def cmd_list(self, rest: str) -> None:
        from wardcat import Wardcat

        what, _, rest = rest.partition(" ")
        if what in ("", "layers"):
            self.cmd_layers("")
            return
        if what != "filters":
            raise ConfigError("usage: list filters [--active|--inactive] | list layers")
        p = _Parser(prog="list filters", allow_abbrev=False)
        side = p.add_mutually_exclusive_group()
        side.add_argument("--active", action="store_true")
        side.add_argument("--inactive", action="store_true")
        ns = p.parse_args(shlex.split(rest))
        support = {layer: Wardcat.supported_entities(layer) for layer in sessions.LAYERS}
        rows = []
        for name in sorted(Wardcat.supported_entities()):
            action = self.state["filters"].get(name)
            if (ns.active and action is None) or (ns.inactive and action is not None):
                continue
            layers = "/".join(layer for layer, names in support.items() if name in names)
            rows.append([name, layers or "-", action or "-"])
        print(table(["Entity", "Layers", "Action"], rows) if rows else "(none)")

    # ---------------------------------------------------------------- layers

    def cmd_layers(self, rest: str) -> None:
        rows = [["regex", "on" if "regex" in self.state["layers"] else "off", "built in"]]
        ner, llm = self.state["ner"], self.state["llm"]
        rows.append(
            [
                "ner",
                "on" if "ner" in self.state["layers"] else "off",
                f"language {ner['language']}" + (f", {ner['size']}" if ner.get("size") else "")
                if ner
                else "not set up",
            ]
        )
        rows.append(
            [
                "llm",
                "on" if "llm" in self.state["layers"] else "off",
                f"{llm['backend']} {llm['model']}"
                + (f" at {llm['base_url']}" if llm.get("base_url") else "")
                if llm
                else "not set up",
            ]
        )
        print(table(["Layer", "State", "Setup"], rows))

    def _add_layer(self, rest: str) -> None:
        ns = _parser(
            "add layer",
            (("layer",), {"choices": list(sessions.LAYERS)}),
            (("--language",), {"help": "ner: the language, e.g. tr"}),
            (("--size",), {"choices": ["sm", "md", "lg", "trf"], "help": "ner: model size"}),
            (("--backend",), {"choices": ["ollama", "vllm", "openai_compatible", "transformers"]}),
            (("--model",), {"help": "llm: the model; without it the screen asks"}),
            (("--base-url",), {"dest": "base_url"}),
            (("--api-key-env",), {"dest": "api_key_env", "metavar": "VAR"}),
            (("--allow-http",), {"action": "store_true", "dest": "allow_http"}),
            (("--adjudicate",), {"action": "store_true"}),
        ).parse_args(shlex.split(rest))
        layer = ns.layer
        ner, llm = self.state["ner"], self.state["llm"]
        reconfigure = (layer == "ner" and ns.language) or (layer == "llm" and ns.model)
        if layer in self.state["layers"] and not reconfigure:
            print(f"layer {layer} is already on")
            return
        if layer == "ner" and ns.language:
            ner = {"language": ns.language, "size": ns.size}
        elif layer == "ner" and ner is None:
            raise ConfigError("the first time, say which language: add layer ner --language tr")
        if layer == "llm" and (ns.model or llm is None):
            llm = self._llm_setup(ns)
            if llm is None:
                print(paint("cancelled", "yellow"))
                return

        def change(state: dict[str, Any]) -> None:
            state["ner"], state["llm"] = ner, llm
            if layer not in state["layers"]:
                state["layers"] = [x for x in sessions.LAYERS if x in {*state["layers"], layer}]

        self._apply(change)
        print(paint(f"layer {layer} on", "green"))

    def _llm_setup(self, ns: argparse.Namespace) -> dict[str, Any] | None:
        backend = ns.backend or (
            "ollama"
            if ns.model
            else self.ask(
                "LLM backend (ollama, vllm, openai_compatible, transformers)", default="ollama"
            )
        )
        if backend not in ("ollama", "vllm", "openai_compatible", "transformers"):
            raise ConfigError(f"unknown backend {backend!r}")
        model = ns.model or self.ask("model, e.g. qwen3:14b")
        if not model:
            return None
        llm: dict[str, Any] = {
            "backend": backend,
            "model": model,
            "base_url": ns.base_url,
            "api_key_env": ns.api_key_env,
            "allow_http": ns.allow_http,
            "adjudicate": ns.adjudicate,
            "key_typed": False,
        }
        if backend == "transformers" or ns.model:
            return llm
        if not ns.base_url:
            port, v1 = _BACKENDS[backend]
            host = self.ask("host or IP", default="localhost")
            port = self.ask("port", default=port)
            llm["base_url"] = f"http://{host}:{port}" + ("/v1" if v1 else "")
            if host not in _LOOPBACK:
                print(
                    paint(
                        "this is a remote host over plain HTTP: texts would travel unencrypted",
                        "yellow",
                    )
                )
                if self.ask("continue? (y/N)", default="n").lower() not in (
                    "y",
                    "yes",
                    "e",
                    "evet",
                ):
                    return None
                llm["allow_http"] = True
        if not ns.api_key_env:
            key = self.ask("API key (blank for none; kept in memory only)", secret=True)
            if key:
                self.api_key = key
                llm["key_typed"] = True
        return llm

    def _remove_layer(self, rest: str) -> None:
        layer = rest.strip().lower()
        if layer not in sessions.LAYERS:
            raise ConfigError("usage: remove layer regex|ner|llm")
        if layer not in self.state["layers"]:
            print(f"layer {layer} is already off")
            return
        if len(self.state["layers"]) == 1:
            raise ConfigError("one layer must stay on")

        def change(state: dict[str, Any]) -> None:
            state["layers"].remove(layer)

        self._apply(change)
        print(paint(f"layer {layer} off", "yellow"))

    # ---------------------------------------------------------------- serve

    def cmd_serve(self, rest: str) -> None:
        if self._server is not None:
            print(f"already serving on 127.0.0.1:{self.port}")
            return
        ns = _parser("serve", (("--port",), {"type": int, "default": 8787})).parse_args(
            shlex.split(rest)
        )
        if not 1 <= ns.port <= 65535:
            raise ConfigError("--port must be between 1 and 65535")
        self._start_server(ns.port)
        print(
            paint(f"serving on http://127.0.0.1:{self.port}", "green")
            + " · loopback only · changes here restart it with the new policy"
        )

    def cmd_stop_serve(self, rest: str) -> None:
        if self._server is None:
            print("not serving")
            return
        self._stop_server()
        print(paint("stopped serving", "yellow"))

    def _start_server(self, port: int) -> None:
        try:
            import uvicorn

            from wardcat.server import ServerConfig, create_app
        except ImportError:
            raise ConfigError('serving needs the service: pip install "wardcat[serve]"') from None
        # The service gets a guard of its own, built from the session as it is
        # now: it is never changed under a request, and a later change here
        # restarts it rather than editing it live.
        _check_port_free(port)
        app = create_app(self._make_guard(), ServerConfig())
        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical", access_log=False)
        )

        def serve() -> None:
            try:
                server.run()
            except BaseException:  # noqa: BLE001 - surfaces below as "did not start"
                pass

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        deadline = time.monotonic() + 5
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not server.started:
            server.should_exit = True
            raise ConfigError(f"the service did not start on port {port}")
        self._server, self._server_thread, self.port = server, thread, port

    def _stop_server(self) -> None:
        if self._server is None:
            return
        self._server.should_exit = True
        if self._server_thread is not None:
            self._server_thread.join(timeout=5)
        self._server = self._server_thread = None
        self.port = None

    def _restart_server(self) -> None:
        port = self.port
        assert port is not None
        self._stop_server()
        self._start_server(port)
        print(paint(f"service on 127.0.0.1:{port} restarted with the new policy", "dim"))

    # ---------------------------------------------------------------- screen

    def cmd_clear(self, rest: str) -> None:
        from wardcat import __version__

        if self.interactive:
            print("\x1b[2J\x1b[H", end="")
            print(banner(__version__))
        print(paint(self.status(), "dim"))

    def cmd_help(self, rest: str) -> None:
        print(help_text())


_COMMANDS: dict[str, Callable[[Shell, str], None]] = {
    "scan": Shell.cmd_scan,
    "add": Shell.cmd_add,
    "remove": Shell.cmd_remove,
    "preset": Shell.cmd_preset,
    "filters": Shell.cmd_filters,
    "list": Shell.cmd_list,
    "layers": Shell.cmd_layers,
    "serve": Shell.cmd_serve,
    "stop-serve": Shell.cmd_stop_serve,
    "clear": Shell.cmd_clear,
    "help": Shell.cmd_help,
}


def _check_port_free(port: int) -> None:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            raise ConfigError(f"port {port} is in use; try serve --port N") from None


def _highlight(result: Any) -> str:
    """The sanitized text with each replacement painted green."""
    text = result.sanitized_text
    spans = sorted(
        (v.sanitized_start, v.sanitized_end)
        for v in result.violations
        if v.replacement is not None and v.sanitized_start is not None
    )
    out, at = [], 0
    for start, end in spans:
        if start < at:
            continue
        out.append(text[at:start])
        out.append(paint(text[start:end], "green"))
        at = end
    out.append(text[at:])
    return "".join(out)


def history_skips(line: str) -> bool:
    """A line the prompt history must not keep: anything that carries scanned text."""
    word = line.strip().lstrip("/").strip().split(" ", 1)[0].lower()
    return word == "scan"


def _prompt_session(shell: Shell) -> Any:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.filters import has_completions
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings

    from wardcat import Wardcat
    from wardcat.presets import PRESETS

    entities = sorted(Wardcat.supported_entities())
    presets = sorted(PRESETS)
    path = sessions.state_dir() / "shell_history"

    class History(FileHistory):
        def store_string(self, string: str) -> None:
            if history_skips(string):
                return
            if not path.exists():
                sessions.write_private(path, "")
            super().store_string(string)

    class Palette(Completer):
        def get_completions(self, document: Any, complete_event: Any) -> Any:
            for value, start, meta in candidates(document.text_before_cursor, entities, presets):
                yield Completion(value, start_position=start, display_meta=meta)

    keys = KeyBindings()

    @keys.add("enter", filter=has_completions)
    def _take(event: Any) -> None:
        # On a `/` line Enter takes the highlighted item, or the first, so the
        # command's arguments can follow. Elsewhere an item Tab has already put
        # on the line runs with it, and an untouched menu is simply closed.
        buffer = event.current_buffer
        state = buffer.complete_state
        palette = buffer.document.text_before_cursor.lstrip().startswith("/")
        chosen = state.current_completion
        if palette:
            chosen = chosen or (state.completions[0] if state.completions else None)
            if chosen is not None and buffer.text.lstrip("/").strip() != chosen.text:
                buffer.apply_completion(chosen)
                return
        buffer.complete_state = None
        buffer.validate_and_handle()

    return PromptSession(
        completer=Palette(),
        complete_while_typing=True,
        history=History(str(path)),
        key_bindings=keys,
    )

"""``wardcat completion SHELL``: print a completion script for bash, zsh or fish.

The script is generated from the parser itself, so it always offers the
commands and options of the installed version, with the fixed choices an
option takes (``--format``, ``--group``, ``--preset``, ...). File names are
left to the shell.

    eval "$(wardcat completion bash)"          # ~/.bashrc
    eval "$(wardcat completion zsh)"           # ~/.zshrc
    wardcat completion fish > ~/.config/fish/completions/wardcat.fish
"""

from __future__ import annotations

import argparse
import shlex


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "completion",
        help="print a shell completion script (bash, zsh, fish)",
        allow_abbrev=False,
    )
    p.add_argument("shell", choices=["bash", "zsh", "fish"])
    p.set_defaults(run=run)


def run(args: argparse.Namespace) -> int:
    from wardcat.cli import EXIT_CLEAN, _parser

    commands = describe(_parser())
    script = {"bash": bash, "zsh": zsh, "fish": fish}[args.shell](commands)
    print(script, end="")
    return EXIT_CLEAN


Options = dict[str, list[str]]  # option string -> its fixed choices ([] when free-form)


def describe(parser: argparse.ArgumentParser) -> dict[str, tuple[str, Options, list[str]]]:
    """``{command: (help, options, positional choices)}`` for every subcommand."""
    known = _known_values()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    helps = {choice.dest: choice.help or "" for choice in sub._choices_actions}
    commands = {}
    for name, command in sub.choices.items():
        options: Options = {}
        positional: list[str] = []
        for action in command._actions:
            choices = [str(c) for c in action.choices or []]
            if action.option_strings:
                for flag in action.option_strings:
                    if flag.startswith("--"):
                        options[flag] = choices or known.get(flag, [])
            else:
                positional.extend(choices)
        commands[name] = (helps.get(name, ""), options, positional)
    return commands


def _known_values() -> Options:
    """Values the parser checks later rather than through ``choices``."""
    from wardcat import entity_groups
    from wardcat.presets import PRESETS

    groups = sorted(
        name.removesuffix("_entities")
        for name in dir(entity_groups)
        if name.endswith("_entities") and callable(getattr(entity_groups, name))
    )
    return {
        "--preset": sorted(PRESETS),
        "--group": groups,
        "--entity": sorted(entity_groups.all_entities()),
    }


def bash(commands: dict[str, tuple[str, Options, list[str]]]) -> str:
    cases = []
    for name, (_, options, positional) in commands.items():
        values = "".join(
            f"                {flag}) COMPREPLY=($(compgen -W {shlex.quote(' '.join(c))}"
            ' -- "$cur")); return;;\n'
            for flag, c in options.items()
            if c
        )
        cases.append(
            f"        {name})\n"
            + (f'            case "$prev" in\n{values}            esac\n' if values else "")
            + f"            words={shlex.quote(' '.join([*options, *positional]))};;\n"
        )
    top = shlex.quote(" ".join(["--version", *commands]))
    return (
        "# wardcat bash completion\n"
        "_wardcat() {\n"
        "    local cur=${COMP_WORDS[COMP_CWORD]} prev=${COMP_WORDS[COMP_CWORD-1]} words\n"
        '    if [ "$COMP_CWORD" -eq 1 ]; then\n'
        f'        COMPREPLY=($(compgen -W {top} -- "$cur")); return\n'
        "    fi\n"
        "    case ${COMP_WORDS[1]} in\n" + "".join(cases) + "        *) return;;\n"
        "    esac\n"
        '    COMPREPLY=($(compgen -W "$words" -- "$cur"))\n'
        "}\n"
        "complete -o default -F _wardcat wardcat\n"
    )


def zsh(commands: dict[str, tuple[str, Options, list[str]]]) -> str:
    # zsh runs the bash script through its bundled bashcompinit layer.
    return (
        "# wardcat zsh completion\n"
        "(( $+functions[compdef] )) || { autoload -U +X compinit && compinit }\n"
        "autoload -U +X bashcompinit && bashcompinit\n" + bash(commands).split("\n", 1)[1]
    )


def fish(commands: dict[str, tuple[str, Options, list[str]]]) -> str:
    lines = [
        "# wardcat fish completion",
        "complete -c wardcat -f -n __fish_use_subcommand -l version",
    ]
    for name, (help_text, options, positional) in commands.items():
        lines.append(
            f"complete -c wardcat -f -n __fish_use_subcommand -a {shlex.quote(name)}"
            f" -d {shlex.quote(help_text)}"
        )
        when = f"-n {shlex.quote(f'__fish_seen_subcommand_from {name}')}"
        if positional:
            lines.append(f"complete -c wardcat {when} -f -a {shlex.quote(' '.join(positional))}")
        for flag, choices in options.items():
            line = f"complete -c wardcat {when} -l {flag[2:]}"
            if choices:
                line += f" -x -a {shlex.quote(' '.join(choices))}"
            lines.append(line)
    return "\n".join(lines) + "\n"

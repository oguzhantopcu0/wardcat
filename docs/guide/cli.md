# Command line

`pip install wardcat` puts a `wardcat` command on the path. It needs nothing the
library does not, and it never prints a value it found: output is the sanitized
text, a PII-free summary, or positions and entity types.

| Command | Does |
|---|---|
| `wardcat` | in a terminal, open the [interactive screen](#interactive-screen) |
| `wardcat scan` | sanitize standard input, files or a JSON Lines dataset |
| `wardcat restore` | put tokenized values back into a model's answer |
| `wardcat check` | find secrets and PII across files, for pre-commit and CI ([guide](ci.md)) |
| `wardcat is-sensitive` | ask the LLM layer whether a text is sensitive at all |
| `wardcat hook claude-code` | keep PII out of a Claude Code session, as a hook |
| `wardcat serve` | run the guard as an HTTP service, with `wardcat[serve]` ([guide](server.md)) |
| `wardcat check-config` | load a policy file through every validation |
| `wardcat entities` | list what can be detected |
| `wardcat presets` | list the presets, or show what one covers and leaves out |
| `wardcat models` | list the NER models per language, or install one |
| `wardcat sessions` | list or delete the screen's saved sessions |
| `wardcat completion SHELL` | print a completion script for bash, zsh or fish |
| `wardcat --version` | print the version |

## Interactive screen

`wardcat` on its own, in a terminal, opens a prompt for trying a policy by
hand: switch layers and filters on and off, load a preset, scan text, and
serve the result while you work.

```text
New session 9ad021ae
session 9ad021ae · layers: regex · 8 filters · type / for the menu, help for every command

wardcat ❯ scan Kartım 4111 1111 1111 1111, TC 10000000146
Kartım [CREDIT_CARD], TC [TC_ID]

Entity       Action  Replacement    Confidence  Layer
───────────  ──────  ─────────────  ──────────  ─────
CREDIT_CARD  redact  [CREDIT_CARD]  1.00        regex
TC_ID        redact  [TC_ID]        1.00        regex

wardcat ❯ preset kvkk
wardcat ❯ add layer ner tr
wardcat ❯ serve --port 8787
```

Type `/` for a menu of commands that narrows as you type, and Tab to complete
commands, entity types and preset names; a status line under the prompt shows
the session, the layers, the filter count and the service. `help` lists every
command:

| Command | Does |
|---|---|
| `scan [TEXT]` | sanitize the text and list what was found — types, actions, replacements, never the values |
| `add filter ENTITY[,ENTITY=ACTION] [--action A]` / `remove filter ENTITY` | look for a type, with an action (default `redact`), or stop |
| `preset NAME` | replace the filters with a preset's |
| `filters`, `list filters [--active\|--inactive]` | the active filters; every type and which layer finds it |
| `add layer regex\|ner\|llm` / `remove layer LAYER` / `layers` | switch detectors; `ner` takes a language the first time (`add layer ner tr`, `add layer ner turkish md`) and turns on `PERSON`, `ORG` and `LOCATION` if no filter uses it; `llm` asks for backend, model, address and key |
| `serve [--port P]` / `stop-serve` | serve the session's policy on 127.0.0.1; a change restarts it with the new policy |
| `clear`, `help`, `quit` | Ctrl-D leaves too; Ctrl-C clears the line |

A new session starts with the regex layer and `CREDIT_CARD`, `CUSTOM_SECRET`,
`EMAIL`, `IBAN`, `JWT`, `PHONE`, `TC_ID` and `VEHICLE_PLATE` redacted. Each
session is saved as you go and can be picked up again:

```bash
wardcat sessions                      # the saved sessions, latest first
wardcat --resume 9ad021ae             # resume one
wardcat --continue                    # resume the latest
wardcat --salt-env WARDCAT_SALT       # salt hash, tokenize and surrogate
```

What stays out of files: a session holds the filters and layer setup, not a
scanned text; the salt is referred to by its variable's name; an LLM API key
comes from `--api-key-env VAR` or a hidden prompt and is kept in memory only, so
a resumed session asks for it again. The prompt history leaves out every `scan`
line. Sessions and history live under `$XDG_STATE_HOME/wardcat`
(`~/.local/state/wardcat`, or `%LOCALAPPDATA%\wardcat` on Windows), readable
by their owner alone.

Outside a terminal — a script, a pipe, CI — `wardcat` with no command prints
its usage and exits 2.

## scan

```bash
# stdin to stdout; each comma-separated item takes its own =ACTION
echo "mail ali@example.com, card 4111 1111 1111 1111" | wardcat scan --entity EMAIL,CREDIT_CARD=mask

# a file, a preset, the PII-free JSON summary written to a file
wardcat scan ticket.txt --preset pci_dss --json --output ticket.json

# whole entity groups, Turkish NER from an installed model, refuse a degraded scan
wardcat scan chunk.txt --group turkish --ner-language tr --strict

# the salt comes from the environment, never from an argument
WARDCAT_SALT=... wardcat scan notes.txt --preset kvkk --salt-env WARDCAT_SALT
```

`--entity` is repeatable and takes comma-separated items; `--entity EMAIL,CREDIT_CARD=mask`
redacts the e-mail with `--action` (default `redact`) and masks the card.
`--group` enables a whole group: `core`, `financial`, `turkish`, `european`,
`uk`, `us`, `network`, `identity` or `all`. Something must be chosen — a
`--preset`, `--entity`, `--group` or `--config` — or the command exits 2.

A `hash`, `tokenize` or `surrogate` action without `--salt-env` still runs, with
a warning: a low-entropy value hashed without a salt can be recovered by brute
force.

The LLM layer takes `--llm MODEL` with `--llm-backend`, `--llm-base-url`,
`--llm-api-key-env VAR` and `--allow-http`; `--adjudicate` lets it confirm the
other layers' candidates. The NER layer takes either `--ner MODEL` (a SpaCy
package) or `--ner-language LANG` (the catalog's model for that language). The
CLI never downloads a model during a scan: install it first, with
`wardcat models pull` or pip.

Detection can be tuned like the library: `--min-confidence 0.8` drops weaker
matches, `--phone-region GB,US` adds national phone formats, `--ner-size md`
picks a larger catalog model for `--ner-language`, `--llm-timeout SECONDS`
bounds each LLM call, `--propagate` masks every later mention of a value found
once, and `--locale tr` shapes surrogates for that language.

### Several files and datasets

```bash
# several files: each sanitized copy goes to the directory under its own name
wardcat scan a.txt b.txt c.txt --entity EMAIL,PHONE --output-dir clean/

# JSON Lines: sanitize one field per line, keep the others
wardcat scan prompts.jsonl --jsonl --field prompt --preset gdpr --output clean.jsonl
```

All texts go through one batch, so the NER layer sees them in a single pass.
The batch fails closed: if any text cannot be scanned — larger than the policy
allows, say — nothing is written and the command exits 2.

## restore

`tokenize` replaces each value with a placeholder a model can carry through its
answer. `--token-map` records what `restore` needs to put the values back:

```bash
wardcat scan prompt.txt --entity EMAIL=tokenize,PERSON=tokenize --ner-language en \
  --salt-env WARDCAT_SALT --token-map map.json > safe-prompt.txt
# ... send safe-prompt.txt to the model, save its answer ...
wardcat restore answer.txt --token-map map.json
```

!!! warning "The token map is raw PII"
    It holds the original values. It is written only when asked for, readable by
    its owner alone, and should be deleted once the answer is restored.

A placeholder that stood for more than one value, or that the map does not know,
is left in place and reported on standard error by type and reason (exit 1).
`--strict` refuses instead (exit 2, nothing printed). `--sources` appends the
list of what was put back.

Input and output are UTF-8 on every platform, Windows consoles and pipes
included. Long options must be written in full; abbreviations are not accepted.

## is-sensitive

```bash
wardcat is-sensitive memo.txt --llm qwen3:14b
# sensitive        (exit 1)
```

It prints `sensitive` or `clean`. The check fails closed: when the backend
cannot be reached, or its circuit is open, the command exits 3 and prints
neither. An input over the size limit is refused (exit 2) rather than judged
in parts.

`--categories` prints `{"sensitive": true, "categories": ["health", "pii"]}`
instead. The model's one-line reason is not printed, since it may quote the
text.

## hook claude-code

`wardcat hook claude-code` is a [Claude Code hook](https://code.claude.com/docs/en/hooks).
It reads each event on standard input and answers in Claude Code's format:

| Event | With a finding |
|---|---|
| `UserPromptSubmit` | the prompt is blocked before it reaches the model; the reason names the entity types. Claude Code cannot rewrite a prompt, so it is not masked and sent on. |
| `PreToolUse` | the tool call is denied — a Bash command, a file a Write would create, a URL. `--tool-decision ask` puts it to you instead. |
| `PostToolUse` | the tool has already run, so its output is replaced with the sanitized text before the model reads it, using each type's `--action` (default `redact`). |

Register it in `.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [{"hooks": [{"type": "command", "timeout": 10,
      "command": "wardcat hook claude-code --entity CUSTOM_SECRET,JWT,CREDIT_CARD,IBAN,TC_ID"}]}],
    "PreToolUse": [{"hooks": [{"type": "command", "timeout": 10,
      "command": "wardcat hook claude-code --entity CUSTOM_SECRET,JWT,CREDIT_CARD,IBAN,TC_ID"}]}],
    "PostToolUse": [{"hooks": [{"type": "command", "timeout": 10,
      "command": "wardcat hook claude-code --entity CUSTOM_SECRET,JWT,CREDIT_CARD,IBAN,TC_ID"}]}]
  }
}
```

The same policy options as `scan` apply; a `--config` file keeps the three
entries short. Pick the types with care: with `EMAIL`, a `git config user.email`
command is denied too. A `matcher` narrows `PreToolUse` and `PostToolUse` to
some tools.

The hook fails closed, event by event: if its input cannot be read or a text
cannot be scanned, the prompt or tool call is blocked (exit 2) and a tool's
output is withheld. A few failures are outside its reach, and Claude Code
lets the event through: a hook that times out, a `wardcat` that is
not on the `PATH`, and — for `PostToolUse` only — an option the command line
rejects. Keep to the regex layer, whose scans take milliseconds, and try the
command once by hand after setting it up:

```bash
echo '{"hook_event_name": "UserPromptSubmit", "prompt": "card 4111 1111 1111 1111"}' \
  | wardcat hook claude-code --entity CREDIT_CARD
```

Images in tool output are passed through unread. Registered for any other event, the hook
reports an error (exit 1) and changes nothing.

## check-config, entities, presets and models

`wardcat check-config policy.yaml` loads the file through every validation the
library has — unknown keys, actions, entity specs, and the ReDoS screen on
custom and denylist patterns — and prints the build warnings a guard would
carry. `wardcat entities [--layer regex|ner|llm]` lists what can be detected.

`wardcat presets` lists the presets; `wardcat presets kvkk` shows the entity
types and actions one enables, what it covers, what it does not, and which layers
it needs. `wardcat models list [--language tr]` shows the catalog's NER models
and which are installed; `wardcat models pull tr_core_news_md` installs one. Only
catalog models with an NER component can be pulled.

## Shell completion

```bash
eval "$(wardcat completion bash)"                               # in ~/.bashrc
eval "$(wardcat completion zsh)"                                # in ~/.zshrc
wardcat completion fish > ~/.config/fish/completions/wardcat.fish
```

The script is generated from the installed version: commands, options, and the
values `--format`, `--group`, `--preset` and `--entity` take.

## Exit codes

| Code | `scan`, `check` | `restore` | `is-sensitive` |
|---|---|---|---|
| `0` | nothing found | every placeholder restored | clean |
| `1` | something was found | some were left in place | sensitive |
| `2` | a configuration, usage or read error | the same, or `--strict` refused | the same |
| `3` | the scan was degraded and `--strict` refused it | — | the backend could not answer |

Errors and warnings go to standard error and never contain a scanned value.

# Command line

`pip install wardcat` puts a `wardcat` command on the path. It needs nothing the
library does not, and it never prints a value it found: output is the sanitized
text, a PII-free summary, or positions and entity types.

| Command | Does |
|---|---|
| `wardcat scan` | sanitize one file or standard input |
| `wardcat check` | find secrets and PII across files, for pre-commit and CI ([guide](ci.md)) |
| `wardcat is-sensitive` | ask the LLM layer whether a text is sensitive at all |
| `wardcat check-config` | load a policy file through every validation |
| `wardcat entities` | list what can be detected |
| `wardcat --version` | print the version |

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
CLI never downloads a model: install it first.

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

## check-config and entities

`wardcat check-config policy.yaml` loads the file through every validation the
library has — unknown keys, actions, entity specs, and the ReDoS screen on
custom and denylist patterns — and prints the build warnings a guard would
carry. `wardcat entities [--layer regex|ner|llm]` lists what can be detected.

## Exit codes

| Code | `scan`, `check` | `is-sensitive` |
|---|---|---|
| `0` | nothing found | clean |
| `1` | something was found | sensitive |
| `2` | a configuration, usage or read error | the same |
| `3` | the scan was degraded and `--strict` refused it | the backend could not answer |

Errors and warnings go to standard error and never contain a scanned value.

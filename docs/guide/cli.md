# Command line

`pip install wardcat` puts a `wardcat` command on the path. It needs nothing the
library does not, and it never prints a value it found: output is the sanitized
text, a PII-free summary, or positions and entity types.

| Command | Does |
|---|---|
| `wardcat scan` | sanitize standard input, files or a JSON Lines dataset |
| `wardcat restore` | put tokenized values back into a model's answer |
| `wardcat check` | find secrets and PII across files, for pre-commit and CI ([guide](ci.md)) |
| `wardcat is-sensitive` | ask the LLM layer whether a text is sensitive at all |
| `wardcat serve` | run the guard as an HTTP service, with `wardcat[serve]` ([guide](server.md)) |
| `wardcat check-config` | load a policy file through every validation |
| `wardcat entities` | list what can be detected |
| `wardcat presets` | list the presets, or show what one covers and leaves out |
| `wardcat models` | list the NER models per language, or install one |
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

## Exit codes

| Code | `scan`, `check` | `restore` | `is-sensitive` |
|---|---|---|---|
| `0` | nothing found | every placeholder restored | clean |
| `1` | something was found | some were left in place | sensitive |
| `2` | a configuration, usage or read error | the same, or `--strict` refused | the same |
| `3` | the scan was degraded and `--strict` refused it | — | the backend could not answer |

Errors and warnings go to standard error and never contain a scanned value.

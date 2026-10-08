# pre-commit and CI

`wardcat check` walks files and directories, reports each finding as
`file:line:col ENTITY`, and exits 1 when anything is found. It never rewrites a
file and never prints a value. That makes it a gate: run it before a commit or
in a pipeline, and a secret or an identity number stops the change.

```bash
wardcat check .                               # everything wardcat knows, recursively
wardcat check src/ --entity CUSTOM_SECRET,JWT,CREDIT_CARD,IBAN,TC_ID
wardcat check . --config policy.yaml --format sarif --output wardcat.sarif
git diff --cached | wardcat check - --stdin-filename staged.diff
```

With no `--entity`, `--group`, `--preset` or `--config`, every type wardcat can
find with the layers you enabled is checked. Directories such as `.git`,
`node_modules` and `.venv` are skipped; `--include` and `--exclude` take globs.

`--jobs N` scans files in N processes. It pays off with the NER layer, where
each file is model-bound; the findings are the same either way. In a pre-commit
hook, which runs on every commit, prefer the regex layer and keep NER for CI.

## Nothing passes unscanned

Every way a file could slip through is loud instead:

- **A file that cannot be read** is an error (exit 2), not a skip.
- **A binary file** (a NUL byte in its first 8 KB, no byte-order mark) is skipped
  and listed on standard error.
- **Encodings.** A UTF-16 or UTF-32 file with a byte-order mark is decoded by it;
  anything else is read as UTF-8 with undecodable bytes replaced, so a card
  number in a Latin-1 file is still found.
- **Large files** are split at line breaks under the policy's own
  `max_text_bytes`. A single line longer than that is an error, because cutting
  it could split a value in two.
- **A policy that enables nothing** is an error.

## Output

| `--format` | Shape |
|---|---|
| `text` (default) | `path:line:col  ENTITY  (confidence)` per finding, a summary on stderr |
| `jsonl` | one JSON object per finding: `file`, `line`, `col`, `entity_type`, `action`, `confidence` |
| `sarif` | SARIF 2.1.0, for GitHub code scanning |

## Baselines

A repository with known, accepted findings — test fixtures, say — records them
once and fails only on new ones:

```bash
wardcat check . --baseline .wardcat-baseline.json --write-baseline   # record, exit 0
wardcat check . --baseline .wardcat-baseline.json                    # report only new ones
```

A baseline entry is `file:line:ENTITY`; no value or hash of a value is stored.
Because it is keyed on the line, an edit that moves a finding makes it new again,
and a second finding of the same type on a recorded line is hidden by the first.

## pre-commit

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/oguzhantopcu0/wardcat
    rev: v1.3.0
    hooks:
      - id: wardcat
        # optional: your own scope or policy
        # args: [--config, policy.yaml]
```

pre-commit installs wardcat without extras (the hook needs none) and passes the
staged files. The default `args` look for secrets, cards, IBANs and national
identity numbers; e-mail addresses, phone numbers and IP addresses are left out
because code uses them legitimately. Tags before `v1.3.0` have no hook.

## GitHub Actions

```yaml
jobs:
  wardcat:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      security-events: write      # only for the SARIF upload
    steps:
      - uses: actions/checkout@v4
      - uses: oguzhantopcu0/wardcat@v1.3.0
        with:
          paths: .
          format: sarif
          output: wardcat.sarif
      - uses: github/codeql-action/upload-sarif@v3
        if: always()              # upload even when findings failed the step
        with:
          sarif_file: wardcat.sarif
```

The action installs wardcat from the same ref you pinned in `uses:`, into its own
virtual environment. Inputs: `paths` (space-separated, default `.`), `entities`
(comma-separated), `format` and `output`.

## Moving from wardcat-cli

The separate `wardcat-cli` package is retired; its `check`, `scan`,
`is-sensitive` and `serve` commands are part of `wardcat` from 1.3.0. If it is installed,
remove it (`pip uninstall wardcat-cli`): both install a `wardcat` command.

| wardcat-cli 0.5 | wardcat 1.3 |
|---|---|
| `--entities A,B` / `-e` | `--entity A,B` |
| `--group` / `-g` | `--group` (repeatable) |
| `--ner` / `--ner-language` | `--ner MODEL` or `--ner-language LANG` |
| `--llm --llm-model M` | `--llm M` |
| `--salt` or an implicit `$WARDCAT_SALT` | `--salt-env WARDCAT_SALT` |
| `--text "…"` | pipe it: `printf '%s' "…" \| wardcat scan` |
| `--format json` (check) | `--format jsonl` |
| `--reveal-raw`, `--fail-on-violation` | removed: values are never printed, and finding something already exits 1 |
| `.wardcat.yaml` found automatically | pass `--config policy.yaml` (the policy YAML format) |
| `interactive`, `sessions`, `install-service`, `serve --docker` | not carried over |
| `serve` | `wardcat serve` with `wardcat[serve]`: no runtime policy changes, a key required off loopback ([HTTP service](server.md)) |

A `.wardcat.yaml` written for wardcat-cli does not load as a policy file: its
`scan:` and `check:` sections become `entities:` entries, and `denylist` and
`allowlist` carry over as they are.

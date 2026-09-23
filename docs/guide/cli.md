# Command line

`pip install wardcat` puts a `wardcat` command on the path. It needs nothing the
library does not, and it never prints a value it found: the output is the
sanitized text, or with `--json` the PII-free `redacted()` dict.

```bash
# stdin to stdout, two entities, the default action is redact
echo "mail ali@example.com" | wardcat scan --entity EMAIL --entity CREDIT_CARD=mask

# a file, a preset, JSON out
wardcat scan ticket.txt --preset pci_dss --json

# a policy file, the NER layer, refuse a degraded scan
wardcat scan chunk.txt --config policy.yaml --ner tr_core_news_lg --strict

# the salt comes from the environment, never from an argument
WARDCAT_SALT=... wardcat scan notes.txt --preset kvkk --salt-env WARDCAT_SALT
```

| Exit code | Meaning |
|---|---|
| `0` | the text is clean |
| `1` | something was found (and replaced in the output) |
| `2` | a configuration or usage error, printed to stderr |
| `3` | the scan was degraded and `--strict` refused it |

`wardcat check-config policy.yaml` loads the file through every validation the
library has — unknown keys, actions, entity specs, and the ReDoS screen on
custom and denylist patterns — and prints the build warnings a guard would
carry. `wardcat entities [--layer regex|ner|llm]` lists what can be detected.

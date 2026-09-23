<p align="center">
  <img src="https://raw.githubusercontent.com/oguzhantopcu0/wardcat/main/docs/assets/logo.png" alt="wardcat" width="240">
</p>

# wardcat

[![CI](https://github.com/oguzhantopcu0/wardcat/actions/workflows/ci.yml/badge.svg)](https://github.com/oguzhantopcu0/wardcat/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/wardcat?label=pypi)](https://pypi.org/project/wardcat/)
[![Release](https://img.shields.io/github/v/release/oguzhantopcu0/wardcat?label=release)](https://github.com/oguzhantopcu0/wardcat/releases)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/oguzhantopcu0/wardcat/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**wardcat finds PII and secrets in text before it reaches an LLM and replaces
them** — checksum-validated regex out of the box, optional SpaCy NER and an
on-prem LLM layer; Turkish, English, German and French. Nothing leaves your
machine.

📖 **Documentation:** <https://docs.wardcat.com>

> **Disclaimer.** wardcat is a **best-effort** PII detector — it does not catch everything (false negatives and positives are expected) and is **not legal advice or a substitute for compliance review** (e.g. GDPR/KVKK); using it does not by itself make a system compliant. Validate it against your own data and requirements. Provided "as is" (MIT — see [LICENSE](https://github.com/oguzhantopcu0/wardcat/blob/main/LICENSE)).

<a name="installation"></a>

## Install

Python 3.11+.

```bash
pip install wardcat              # regex layer + Ollama / OpenAI-compatible LLM backend
pip install "wardcat[ner]"       # + SpaCy NER (names, organisations, places)
pip install "wardcat[phone]"     # + national phone formats worldwide (libphonenumber)
pip install "wardcat[all]"       # everything, including the in-process Transformers backend
```

Details and SpaCy models: [Installation](https://docs.wardcat.com/installation/).

<a name="quick-start"></a>

## First scan

Detection is opt-in: a bare `Wardcat()` detects nothing. Name the entities you
care about and the action for each. This runs offline, with no model:

```python
from wardcat import Wardcat, Entity, Action

guard = (
    Wardcat(salt="change-me")                                   # your app supplies the salt
    .add_entities([Entity.TC_ID, Entity.CREDIT_CARD], action=Action.HASH)
    .add_entities([Entity.IBAN, Entity.PHONE], action=Action.REDACT)
    .add_entity(Entity.EMAIL, Action.MASK)
)

text = "Müşteri TC 10000000146, IBAN TR33 0006 1005 1978 6457 8413 26, tel +90 532 123 45 67. Card 4111 1111 1111 1111, contact ayse.kaya@example.com"
result = guard.scan(text)
print(result.sanitized_text)
# Müşteri TC [TC_ID:80767a764888c13d], IBAN [IBAN], tel [PHONE]. Card [CREDIT_CARD:e2292d1b12e86559], contact a********@example.com
print(result.is_clean, [v.entity_type for v in result.violations])
# False ['TC_ID', 'IBAN', 'PHONE', 'CREDIT_CARD', 'EMAIL']
```

The [Quickstart](https://docs.wardcat.com/quickstart/) adds the YAML form,
entity groups and batches. Add `.with_ner(language="tr")` for names and
`.with_llm(model=...)` for contextual and semantic detection — both are on the
[Detection layers](https://docs.wardcat.com/guide/layers/) page, with the async API.

The same scan from a shell:

```bash
echo "mail ali@example.com, card 4111 1111 1111 1111" | wardcat scan --entity EMAIL --entity CREDIT_CARD=mask
# mail [EMAIL], card ************1111
```

## Why wardcat

- **A match is proof, not a guess.** Cards (Luhn, Troy included), IBANs (mod-97),
  TC IDs, Bitcoin addresses, NHS and ABA numbers, IMEIs, Spanish, French, Dutch
  and Polish national IDs are checksum-verified before they are flagged. Weak evidence is scored
  below a [confidence floor](https://docs.wardcat.com/guide/configuration/#confidence-floor)
  and left alone until you lower it.
- **Three layers, one policy.** Regex is always on; SpaCy NER and an on-prem LLM
  (Ollama, vLLM, any OpenAI-compatible server, or in-process Transformers) are
  opt-in. A deterministic regex span always wins an overlap, and the LLM can
  [adjudicate](https://docs.wardcat.com/guide/layers/#ensemble-adjudication)
  the other layers' candidates in one call.
- **Six actions, two of them reversible.** `warn`, `hash` (salted SHA-256),
  `redact`, `mask`, `tokenize` and `surrogate` (a realistic stand-in of the
  same shape). Tokens and surrogates come back with
  [`restore()`](https://docs.wardcat.com/guide/reversible/) after the model has answered.
- **Turkish first, then Europe.** TC ID, IBAN, Vergi No, plates and postcodes
  beside SSN, NIN, DNI/NIE, INSEE, BSN, PESEL, UK postcodes and ZIP+4; names,
  addresses, birth dates and phone numbers in TR/EN/DE/FR; API keys for a dozen
  providers, PEM blocks and credentials written into a sentence. Lookalike
  characters (Cyrillic, fullwidth digits) are folded before matching.
- **It never fails quietly.** A layer that cannot run is reported on
  `result.warnings`, or refused outright with
  [`with_strict()`](https://docs.wardcat.com/guide/configuration/#strict-mode-refuse-a-degraded-scan);
  a dead backend trips a circuit breaker instead of costing every scan its
  timeout. The library reads no environment variables and its log lines never
  carry a value.

With the LLM layer on, [`is_sensitive()`](https://docs.wardcat.com/guide/layers/#semantic-sensitivity-gate-is_sensitive)
is a yes/no gate for text with no typed entity — a leaked forecast, a deal
term — and `classify()` says which kind. A [reproducible benchmark](https://github.com/oguzhantopcu0/wardcat/tree/main/benchmarks)
scores wardcat against Microsoft Presidio on public corpora.

<a name="supported-entity-types"></a>
<a name="known-limitations"></a>

## Learn more

- [Quickstart](https://docs.wardcat.com/quickstart/) — programmatic and YAML APIs, entity groups, batches, the examples.
- [Detection layers](https://docs.wardcat.com/guide/layers/) — regex, NER, the LLM backends, adjudication, the semantic gate.
- [Entity types](https://docs.wardcat.com/reference/entities/) — every entity, its default action and which layer finds it.
- [Configuration & policy](https://docs.wardcat.com/guide/configuration/) — the confidence floor, propagation, allow/deny lists, strict mode, the circuit breaker, YAML reference.
- [Presets](https://docs.wardcat.com/guide/presets/) — `kvkk`, `gdpr`, `pci_dss`, `hipaa_lite`, `secrets_only` as starting policies.
- [Command line](https://docs.wardcat.com/guide/cli/) — `wardcat scan`, `check-config`, `entities`.
- [Reversible masking](https://docs.wardcat.com/guide/reversible/) — tokens, surrogates, `restore()` and the reverse map.
- [Extending](https://docs.wardcat.com/guide/extending/) — custom actions and detectors.
- [MCP server](https://docs.wardcat.com/guide/mcp-server/) — wardcat as a tool for an agent.
- [Known limitations](https://docs.wardcat.com/guide/limitations/) — what is not caught, and what to do about it.
- [API reference](https://docs.wardcat.com/reference/wardcat/) · [Examples](https://github.com/oguzhantopcu0/wardcat/tree/main/examples) · [Changelog](https://github.com/oguzhantopcu0/wardcat/blob/main/CHANGELOG.md)

<a name="security"></a>

## Security

Sensitive values are replaced with `[TYPE:16hex]` by default. **Deterministic
hashing is not anonymization:** the same value always hashes to the same token,
and because SHA-256 is fast, low-entropy PII (phone numbers, SSNs, TC IDs) can be
recovered by brute force by anyone who has the salt and knows the value format.
Treat hashed output as *pseudonymized*: keep the salt secret, and use
`redact`/`mask` when you need values that cannot be reversed.

```python
import os
from wardcat import Wardcat

guard = Wardcat(salt=os.environ["WARDCAT_SALT"]).add_entity("CREDIT_CARD", "hash")
```

Never hardcode the salt in source code or config files. `ScanResult.original_text`
and `violations[].original` contain raw PII; use `result.redacted()` for logs and
API responses. Inputs exceeding **500 KB** raise a `ValueError` — split large
documents into smaller chunks before scanning. Plaintext HTTP to a remote LLM
backend is blocked (loopback is fine); the LLM layer is best-effort against
prompt injection, so keep regex and NER in front of it.

The [security guide](https://docs.wardcat.com/guide/security/) has the details.
To report a vulnerability, follow [SECURITY.md](https://github.com/oguzhantopcu0/wardcat/blob/main/SECURITY.md).

## Contributing

Bug reports and pull requests are welcome on
[GitHub](https://github.com/oguzhantopcu0/wardcat/issues).
[CONTRIBUTING.md](https://github.com/oguzhantopcu0/wardcat/blob/main/CONTRIBUTING.md)
covers the development setup, the quality gates and how to add an entity type.

## License

MIT — see [LICENSE](https://github.com/oguzhantopcu0/wardcat/blob/main/LICENSE).

# Security Policy

`wardcat` is a PII-detection and anonymization library, so security issues in it
can directly affect the confidentiality of the data it processes. Please treat
reports seriously and privately.

## Reporting a vulnerability

**Do not open a public issue for a security vulnerability.** Instead, report it
privately via GitHub's [Security Advisories](https://github.com/oguzhantopcu0/wardcat/security/advisories/new)
("Report a vulnerability"), or by email to the maintainer listed in
`pyproject.toml`.

Please include:

- a description of the issue and its impact,
- steps to reproduce (a minimal snippet is ideal), and
- the `wardcat` version and Python version.

You can expect an acknowledgement within a few days and a fix or mitigation plan
for confirmed issues.

## Supported versions

Being pre-1.0, only the latest released `0.x` line receives security fixes.

## Scope & known limitations

Some behaviours are **inherent to the design**, not vulnerabilities — but you
should understand them when relying on wardcat:

- **Hashing is pseudonymization, not anonymization.** Deterministic salted
  SHA-256 lets low-entropy PII (phone numbers, national IDs) be brute-forced by
  anyone holding the salt. Keep the salt secret; use `redact`/`mask` when a value
  must be irreversible. Never commit a salt.
- **The LLM layer can be prompt-injected.** Scanned text is interpolated into the
  prompt, so adversarial input may suppress LLM detections. Treat the LLM layer
  as a best-effort supplement to the deterministic regex layer, not the primary
  mechanism, in adversarial settings.
- **On-prem only.** wardcat never sends PII off-host on its own; plaintext HTTP
  to a *remote* LLM backend is blocked by default. If you point it at a hosted
  endpoint, PII leaves your infrastructure — that is your choice, not a default.

Reports that improve any of the above (e.g. a ReDoS in a built-in pattern, a way
to bypass the HTTP-to-remote guard, or a hash-handling flaw) are in scope.

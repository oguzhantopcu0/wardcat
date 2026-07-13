# MCP server

[**wardcat-mcp**](https://github.com/oguzhantopcu0/wardcat-mcp) is a companion
[Model Context Protocol](https://modelcontextprotocol.io) server that exposes
wardcat's on-prem PII detection as tools any agent can call — Claude Desktop,
Cursor, Cline, a self-hosted bot, or a RAG pipeline. Use it as a **guardrail**:
sanitize inputs before they reach an LLM, or gate them with a semantic
"is this sensitive?" check — without writing any wardcat code.

!!! info "Runs locally, stays local"
    The server runs on your machine over **stdio**; the text, the models, and
    all detection stay on-prem — nothing is sent anywhere. The client spawns it
    as a local subprocess next to your Ollama/vLLM, so PII never leaves the
    machine.

## Tools

| Tool | Description |
|------|-------------|
| `scan(text, entities=None)` | Detect PII and return the **sanitized text** plus a PII-free summary (entity types, actions, confidence — never the raw values), using the server's configured action. |
| `redact(text, action, entities=None)` | Like `scan`, but you choose the **action per call**: `redact` drops the value (`[EMAIL]`), `mask` keeps a hint (`b***@example.com`, last-4 of a card), `hash` gives a stable salted pseudonym (`[EMAIL:3245e00b…]`), `warn` leaves the text untouched but still reports what was found. Defaults to `WARDCAT_ACTION`. |
| `is_sensitive(text)` | Holistic LLM yes/no on whether the text contains sensitive information. Requires the [LLM layer](layers.md) (`WARDCAT_LLM_MODEL`). |

Both `scan` and `redact` accept an optional `entities` list to narrow a single
call to a subset of the server's enabled types (e.g. `["EMAIL", "IBAN"]`);
requesting a type the server didn't enable is an error rather than a silent
no-op.

## Install & run

Not on PyPI yet — run from source (wardcat itself is source-only for now):

```bash
uvx --from git+https://github.com/oguzhantopcu0/wardcat-mcp.git wardcat-mcp
# or, cloned locally:
uv run wardcat-mcp
```

## Add it to an MCP client

Claude Desktop (`claude_desktop_config.json`), Cursor, Cline, Zed, etc.:

```json
{
  "mcpServers": {
    "wardcat": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/oguzhantopcu0/wardcat-mcp.git", "wardcat-mcp"],
      "env": {
        "WARDCAT_SALT": "your-secret-salt",
        "WARDCAT_ACTION": "redact",
        "WARDCAT_LLM_MODEL": "llama3.2:3b"
      }
    }
  }
}
```

## Configuration

The server is configured once at startup via environment variables (mirroring
the library's [configuration & policy](configuration.md) options):

| Variable | Default | Meaning |
|----------|---------|---------|
| `WARDCAT_SALT` | `""` | Hashing salt (required for the `hash` action). |
| `WARDCAT_ENTITIES` | broad structural + name set | Comma-separated entity types to enable. Unknown types are ignored **with a warning**, not silently. |
| `WARDCAT_ACTION` | `redact` | `warn` \| `hash` \| `redact` \| `mask`. |
| `WARDCAT_SPACY_MODEL` | — | Enable [SpaCy NER](layers.md) with this model (needs the `ner` extra). |
| `WARDCAT_LLM_MODEL` | — | Enable the on-prem [LLM layer](layers.md) via Ollama (e.g. `llama3.2:3b`). |
| `WARDCAT_LLM_BASE_URL` | `http://localhost:11434` | Ollama endpoint. |

By default only deterministic, regex-detectable structural PII is enabled
(EMAIL, PHONE, CREDIT_CARD, IBAN, TC_ID, IP, JWT, …); name entities
(PERSON/ORG/ADDRESS) are added only when a NER or LLM layer is configured, since
they need a model to detect.

## Disclaimer

Like the library, wardcat-mcp is a **best-effort** PII detector — it does not
catch everything and is **not legal advice or a substitute for compliance
review** (e.g. GDPR/KVKK). See the [security guide](security.md) and validate it
against your own data.

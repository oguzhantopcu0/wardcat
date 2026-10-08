# HTTP service

`wardcat serve` runs a guard behind an HTTP endpoint any language can call. It
needs the `serve` extra:

```bash
pip install "wardcat[serve]"
wardcat serve --preset kvkk --salt-env WARDCAT_SALT            # 127.0.0.1:8787, no key
WARDCAT_API_KEY=... wardcat serve --config policy.yaml --host 0.0.0.0
```

The guard takes the same options as `wardcat scan` (`--config`, `--preset`,
`--entity`, `--group`, `--ner`, `--llm`, `--strict`, `--salt-env`). The policy is
fixed when the service starts; nothing over HTTP can change it. Restart to
change it.

## Endpoints

| Method and path | Key | Body | Answer |
|---|---|---|---|
| `GET /healthz` | no | — | `{"status": "ok"}` |
| `GET /readyz` | no | — | `{"status": "ready"}`, or 503 `degraded` when a layer failed under `--strict` |
| `GET /info` | yes | — | enabled entity types with their actions, active layers, build warnings |
| `POST /scan` | yes | `{"text": "..."}` | the `redacted()` dict: `sanitized_text`, `is_clean`, `violations` (types, offsets, actions — no values), `warnings`, `scan_error` |
| `POST /scan` (batch) | yes | `{"texts": ["...", ...]}`, at most 1000 | `{"results": [...]}`, one `redacted()` dict per text, in order; if any text cannot be scanned the whole batch is 413 |
| `POST /classify` | yes | `{"text": "..."}` | `{"sensitive": true, "categories": ["health"]}`; needs `--llm`, otherwise 409; the model's reason is not returned |
| `POST /is-sensitive` | yes | `{"text": "..."}` | `{"sensitive": true}`; needs `--llm`, otherwise 409 |
| `GET /metrics` | yes | — | request counts by path and status, Prometheus text format |

Send the key as `Authorization: Bearer <key>` or `x-api-key: <key>`. POST bodies
must be `application/json`.

```bash
curl -s localhost:8787/scan -H "x-api-key: $WARDCAT_API_KEY" \
  -H 'content-type: application/json' -d '{"text": "mail ali@example.com"}'
```

## Security defaults

- **The key comes from the environment** (`--api-key-env`, default
  `WARDCAT_API_KEY`), never the command line. It must be at least 16 characters.
- **No key, loopback only.** Without a key the service refuses to listen on
  anything but a loopback address, and answers only requests whose `Host`
  header is a loopback name (`127.0.0.1`, `[::1]`, `localhost`), so a web page
  cannot reach it through DNS rebinding.
- **No cross-origin access by default.** A request carrying an `Origin` header
  is refused unless `--cors-origin` names it; `*` is not accepted. JSON bodies
  force a browser to ask first.
- **Bounded work.** A body larger than the policy's `max_text_bytes` (plus room
  for the JSON around it) is refused with 413 while it is being read.
  `--max-concurrency` (default 8) caps requests in flight, and more get 503
  `busy`. `--request-timeout` (default 30 s) answers a slow one with 503 `timeout`.
- **No values in answers or logs.** Error bodies are fixed codes:
  `bad_request`, `unsupported_media_type`, `too_large`, `unauthorized`,
  `bad_host`, `cross_origin`, `busy`, `timeout`, `degraded`, `llm_not_configured`,
  `llm_unavailable` or `internal_error`. The access log records method, path,
  status and duration; no query string, no body.

If you ran wardcat-cli's `serve`: its `WARDCAT_AUTH_TOKEN` is not read any more,
and the service refuses to start while that variable is set and
`WARDCAT_API_KEY` is not. `/health` is now `/healthz`, and `POST /layers` and
`POST /filters` are gone.

## Container

The repository's `Dockerfile` builds a non-root image with the regex layer:

```bash
docker build -t wardcat .
docker run --rm -p 8787:8787 -e WARDCAT_API_KEY -e WARDCAT_SALT \
  -v "$PWD/policy.yaml:/etc/wardcat/policy.yaml:ro" \
  wardcat --config /etc/wardcat/policy.yaml --salt-env WARDCAT_SALT
```

The container listens on `0.0.0.0`, so it will not start without
`WARDCAT_API_KEY`. For NER, build from this image and install `wardcat[ner]`
with the SpaCy model you need.

## Running it as a service

A systemd unit, with the key and salt in a root-only file:

```ini
# /etc/systemd/system/wardcat.service
[Unit]
Description=wardcat PII service
After=network-online.target

[Service]
EnvironmentFile=/etc/wardcat/env          # WARDCAT_API_KEY=… WARDCAT_SALT=…  (chmod 600)
ExecStart=/opt/wardcat/bin/wardcat serve --config /etc/wardcat/policy.yaml --salt-env WARDCAT_SALT
DynamicUser=yes
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

With several workers or replicas, run one service per process: each loads its
own NER model, and nothing is shared between them.

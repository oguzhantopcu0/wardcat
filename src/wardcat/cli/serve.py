"""``wardcat serve``: run the guard as an HTTP service. Needs ``wardcat[serve]``.

The API key comes from an environment variable, never the command line. Without
one the service only runs on a loopback address; anything else exits 2 before
it listens. The policy is fixed at start; restart to change it.
"""

from __future__ import annotations

import argparse
import os
import sys

from wardcat.cli._guard_args import add_guard_args, build_guard
from wardcat.exceptions import ConfigError

_INSTALL_HINT = "wardcat serve needs the serve extra: pip install 'wardcat[serve]'"


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("serve", help="run the guard as an HTTP service", allow_abbrev=False)
    add_guard_args(p)
    net = p.add_argument_group("service")
    net.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    net.add_argument("--port", type=int, default=8787, help="port (default: 8787)")
    net.add_argument(
        "--api-key-env",
        default="WARDCAT_API_KEY",
        metavar="VAR",
        help="environment variable holding the API key (default: WARDCAT_API_KEY); "
        "required off loopback",
    )
    net.add_argument(
        "--cors-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help="allow a browser origin; repeatable. '*' is refused",
    )
    net.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        metavar="N",
        help="requests handled at once; more get 503 (default: 8)",
    )
    net.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="a request that takes longer gets 503 (default: 30)",
    )
    net.add_argument(
        "--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"]
    )
    p.set_defaults(run=run)


def _api_key(args: argparse.Namespace) -> str | None:
    key = os.environ.get(args.api_key_env, "")
    if key and len(key) < 16:
        raise ConfigError(f"{args.api_key_env} is shorter than 16 characters; use a longer key")
    if not key and os.environ.get("WARDCAT_AUTH_TOKEN"):
        # wardcat-cli read WARDCAT_AUTH_TOKEN. Starting without a key because the
        # name changed would silently drop authentication.
        raise ConfigError(
            f"WARDCAT_AUTH_TOKEN is set but {args.api_key_env} is not: wardcat serve "
            f"reads the key from {args.api_key_env}; move it there"
        )
    return key or None


def run(args: argparse.Namespace) -> int:
    from wardcat.cli import EXIT_CLEAN, EXIT_CONFIG

    try:
        import uvicorn

        from wardcat.server import ServerConfig, create_app
        from wardcat.server._hosts import address_is_loopback
    except ImportError:
        print(f"error: {_INSTALL_HINT}", file=sys.stderr)
        return EXIT_CONFIG

    if "*" in args.cors_origin:
        raise ConfigError(
            "--cors-origin '*' is refused; list the origins that may call the service"
        )
    if args.max_concurrency < 1 or args.request_timeout <= 0:
        raise ConfigError("--max-concurrency and --request-timeout must be positive")
    key = _api_key(args)
    if key is None and not address_is_loopback(args.host):
        raise ConfigError(
            f"refusing to listen on {args.host} without an API key: set {args.api_key_env} "
            "(or bind to 127.0.0.1)"
        )
    guard = build_guard(args)
    host = "127.0.0.1" if args.host.strip().lower() == "localhost" else args.host
    app = create_app(
        guard,
        ServerConfig(
            api_key=key,
            cors_origins=tuple(args.cors_origin),
            max_concurrency=args.max_concurrency,
            request_timeout=args.request_timeout,
        ),
    )
    _log_requests(args.log_level)
    print(
        f"wardcat serving on http://{host}:{args.port} "
        f"({'API key required' if key else 'loopback only, no key'})",
        file=sys.stderr,
    )
    uvicorn.run(
        app,
        host=host,
        port=args.port,
        log_level=args.log_level,
        access_log=False,  # our own log has method, path, status, duration; no query
        server_header=False,
    )
    return EXIT_CLEAN


def _log_requests(level: str) -> None:
    """Send the service's access lines (method, path, status, ms) to stderr."""
    import logging

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s:     %(message)s"))
    log = logging.getLogger("wardcat.server")
    log.addHandler(handler)
    log.setLevel(level.upper())
    log.propagate = False

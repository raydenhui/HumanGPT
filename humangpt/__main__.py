"""Entry point: ``python -m humangpt`` and the ``humangpt`` console script."""

from __future__ import annotations

import argparse
import sys

import uvicorn

from .config import load_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="humangpt",
        description="Human-in-the-loop OpenAI-compatible mock endpoint.",
    )
    parser.add_argument("--host", default=None, help="bind host (default: HOST env / 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="bind port (default: PORT env / 8000)")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="uvicorn workers (MUST be 1: parked requests live in-process)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="path to the .env file (default: ./.env)",
    )
    args = parser.parse_args(argv)

    if args.workers != 1:
        parser.error("HumanGPT supports exactly one uvicorn worker; "
                     "parked client requests live in the worker process.")

    from .app import assert_single_worker

    assert_single_worker()

    from .app import create_app

    settings = load_settings(
        dotenv_path=args.env_file and __import__("pathlib").Path(args.env_file)
    )
    host = args.host or settings.host
    port = args.port or settings.port

    print(
        f"HumanGPT listening on http://{host}:{port}\n"
        f"  OpenAI-compatible API : http://{host}:{port}/v1\n"
        f"  Operator UI            : http://{host}:{port}/\n"
        f"  API key mode           : {settings.api_key_mode}\n"
        f"  DB                     : {settings.db_path}\n"
    )

    # Build the app object *in this process* so the Settings resolved from the
    # CLI/`.env` file are the ones actually served (a factory string would
    # re-import create_app() with no arguments and lose them). The signal hook
    # interrupts parked requests before uvicorn's connection-wait on shutdown.
    app = create_app(settings=settings, install_signal_hook=True)

    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=1,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

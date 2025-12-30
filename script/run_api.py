from __future__ import annotations

"""Entry script for running the Loreley read-only UI API (FastAPI).

Usage (with uv):

    uv run python script/run_api.py
"""

import argparse
import sys
from typing import Sequence

from rich.console import Console

from loreley.config import get_settings
from loreley.entrypoints import configure_process_logging, run_api

console = Console()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Loreley read-only UI API (FastAPI).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument("--log-level", dest="log_level", help="Override Settings.log_level.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only).")
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip preflight validation.",
    )
    parser.add_argument(
        "--preflight-timeout-seconds",
        type=float,
        default=2.0,
        help="Network timeout used for DB connectivity checks.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        settings = get_settings()
    except Exception as exc:  # pragma: no cover
        console.log(
            "[bold red]Invalid Loreley configuration[/] "
            f"reason={exc}. Set required environment variables and try again."
        )
        return 1

    try:
        configure_process_logging(
            settings=settings,
            console=console,
            role="ui_api",
            override_level=args.log_level,
        )
    except ValueError as exc:
        console.log("[bold red]Invalid log level[/] reason={}".format(exc))
        return 1

    return int(
        run_api(
            settings=settings,
            console=console,
            host=str(args.host),
            port=int(args.port),
            reload=bool(args.reload),
            preflight=not bool(args.no_preflight),
            preflight_timeout_seconds=float(args.preflight_timeout_seconds),
            uvicorn_log_level=(str(args.log_level) if args.log_level else None),
        )
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())



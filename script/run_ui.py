from __future__ import annotations

"""Entry script for running the Loreley Streamlit UI.

Usage (with uv):

    uv run python script/run_ui.py
"""

import argparse
import os
import sys
from typing import Sequence

from rich.console import Console

from loreley.config import get_settings
from loreley.entrypoints import _coerce_exit_code, configure_process_logging, run_ui

console = Console()

__all__ = ["main", "_coerce_exit_code"]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Loreley Streamlit UI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--api-base-url",
        default=os.getenv("LORELEY_UI_API_BASE_URL", "http://127.0.0.1:8000"),
        help="Base URL of the Loreley UI API.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Streamlit bind host.")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit bind port.")
    parser.add_argument("--headless", action="store_true", help="Run without opening a browser.")
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip preflight validation.",
    )
    parser.add_argument(
        "--preflight-timeout-seconds",
        type=float,
        default=2.0,
        help="Network timeout used for preflight checks.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)

    try:
        settings = get_settings()
    except Exception as exc:  # pragma: no cover - defensive
        console.log(
            "[bold red]Invalid Loreley configuration[/] "
            f"reason={exc}. Set required environment variables and try again."
        )
        return 1

    try:
        configure_process_logging(
            settings=settings,
            console=console,
            role="ui",
            override_level=None,
        )
    except ValueError as exc:
        console.log("[bold red]Invalid log level[/] reason={}".format(exc))
        return 1

    return int(
        run_ui(
            settings=settings,
            console=console,
            api_base_url=str(args.api_base_url),
            host=str(args.host),
            port=int(args.port),
            headless=bool(args.headless),
            preflight=not bool(args.no_preflight),
            preflight_timeout_seconds=float(args.preflight_timeout_seconds),
        )
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())



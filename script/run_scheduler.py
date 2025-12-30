from __future__ import annotations

"""Entry script for running the Loreley evolution scheduler.

This is a thin wrapper around ``loreley.scheduler.main`` that:

- Exposes a small CLI so ``--help`` works without a configured environment.
- Initialises application settings.
- Configures Loguru logging level based on ``Settings.log_level`` and routes
  standard-library logging (used by Dramatiq) through Loguru.
- Delegates CLI parsing and control flow to ``loreley.scheduler.main.main``.

Usage (with uv):

    uv run python script/run_scheduler.py            # continuous loop
    uv run python script/run_scheduler.py --once    # single tick then exit
"""

import argparse
import sys
from typing import Sequence

from rich.console import Console

from loreley.config import get_settings
from loreley.entrypoints import configure_process_logging, run_scheduler

console = Console()


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return a minimal CLI parser and pass through unknown args to the scheduler."""

    parser = argparse.ArgumentParser(
        description="Run the Loreley evolution scheduler.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        help="Override Settings.log_level for this invocation (e.g. DEBUG, INFO).",
    )
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip preflight validation.",
    )
    parser.add_argument(
        "--preflight-timeout-seconds",
        type=float,
        default=2.0,
        help="Network timeout used for DB/Redis connectivity checks.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the scheduler wrapper."""

    parser = _build_arg_parser()
    args, forwarded = parser.parse_known_args(list(argv) if argv is not None else None)

    try:
        settings = get_settings()
    except Exception as exc:  # pragma: no cover - defensive
        console.log(
            "[bold red]Invalid Loreley configuration[/] "
            f"reason={exc}. Use --help for usage and set required environment variables."
        )
        return 1

    try:
        configure_process_logging(
            settings=settings,
            console=console,
            role="scheduler",
            override_level=args.log_level,
        )
    except ValueError as exc:
        console.log("[bold red]Invalid log level[/] reason={}".format(exc))
        return 1

    return int(
        run_scheduler(
            settings=settings,
            console=console,
            argv=list(forwarded),
            preflight=not bool(args.no_preflight),
            preflight_timeout_seconds=float(args.preflight_timeout_seconds),
        )
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())



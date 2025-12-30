from __future__ import annotations

"""Entry script for running the Loreley evolution worker.

This script:

- Parses a small CLI so ``--help`` works even without environment variables.
- Loads application settings and configures Loguru logging, including routing
  standard-library logging (used by Dramatiq) through Loguru.
- Lazily initialises the Dramatiq Redis broker defined in
  ``loreley.tasks.broker`` and imports ``loreley.tasks.workers`` so that the
  ``run_evolution_job`` actor is registered.
- Starts a single Dramatiq worker bound to the configured queue using a
  single-threaded worker pool.

Typical usage (with uv):

    uv run python script/run_worker.py
"""

import argparse
import sys
from typing import Sequence

from loguru import logger
from rich.console import Console

from loreley.config import Settings, get_settings
from loreley.entrypoints import configure_process_logging, run_worker

console = Console()
log = logger.bind(module="script.run_worker")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return a minimal CLI parser so users can ask for help without config."""

    parser = argparse.ArgumentParser(
        description="Run the Loreley evolution worker (single-threaded Dramatiq consumer).",
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


def main(_argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the evolution worker wrapper."""

    parser = _build_arg_parser()
    args = parser.parse_args(list(_argv) if _argv is not None else None)

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
            role="worker",
            override_level=args.log_level,
        )
    except ValueError as exc:
        console.log("[bold red]Invalid log level[/] reason={}".format(exc))
        return 1
    return int(
        run_worker(
            settings=settings,
            console=console,
            preflight=not bool(args.no_preflight),
            preflight_timeout_seconds=float(args.preflight_timeout_seconds),
        )
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())



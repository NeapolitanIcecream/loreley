from __future__ import annotations

"""Entry script for running the Loreley evolution scheduler.

This is a thin wrapper around ``app.scheduler.main`` that:

- Initialises application settings.
- Configures Loguru logging level based on ``Settings.log_level``.
- Delegates CLI parsing and control flow to ``app.scheduler.main.main``.

Usage (with uv):

    uv run python script/run_scheduler.py            # continuous loop
    uv run python script/run_scheduler.py -- --once # single tick then exit
"""

import sys
from typing import Sequence

from loguru import logger
from rich.console import Console

from app.config import get_settings
from app.scheduler.main import main as scheduler_main

console = Console()


def _configure_logging() -> None:
    """Configure Loguru using application settings.

    This keeps logging behaviour consistent across entrypoints.
    """

    settings = get_settings()
    level = (settings.log_level or "INFO").upper()

    # Reset default sinks and install a simple stderr sink.
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        backtrace=False,
        diagnose=False,
    )
    logger.bind(module="script.run_scheduler").info(
        "Scheduler logging initialised at level {}", level
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the scheduler wrapper."""

    _configure_logging()
    # Delegate argument parsing and control flow to app.scheduler.main.
    return int(scheduler_main(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())



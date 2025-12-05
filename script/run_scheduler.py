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
import logging
from datetime import datetime
import sys
from pathlib import Path
from typing import Sequence

from loguru import logger
from rich.console import Console

from loreley.config import Settings, get_settings

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
    return parser


class _LoguruInterceptHandler(logging.Handler):
    """Bridge standard-library logging records into Loguru."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - thin wrapper
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        logger.opt(exception=record.exc_info).log(level, record.getMessage())


def _configure_stdlib_logging(level: str) -> None:
    """Route stdlib logging (including Dramatiq) through Loguru."""

    handler: logging.Handler = _LoguruInterceptHandler()

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    dramatiq_logger = logging.getLogger("dramatiq")
    dramatiq_logger.handlers.clear()
    dramatiq_logger.propagate = True
    dramatiq_logger.setLevel(level)

    logging.captureWarnings(True)


def _resolve_logs_dir(settings: Settings, role: str) -> Path:
    """Return the log directory for the given role, creating it if needed."""

    if settings.logs_base_dir:
        base_dir = Path(settings.logs_base_dir).expanduser()
    else:
        base_dir = Path.cwd()

    logs_root = base_dir / "logs"
    log_dir = logs_root / role
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _configure_logging(settings: Settings, *, override_level: str | None = None) -> None:
    """Configure Loguru and bridge stdlib logging using application settings."""

    level = (override_level or settings.log_level or "INFO").upper()
    try:
        logger.level(level)
    except ValueError as exc:
        raise ValueError(
            f"Invalid log level {level!r}; expected one of TRACE/DEBUG/INFO/SUCCESS/WARNING/ERROR/CRITICAL."
        ) from exc

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        backtrace=False,
        diagnose=False,
    )

    logs_dir = _resolve_logs_dir(settings, role="scheduler")
    log_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = logs_dir / f"scheduler-{log_timestamp}.log"
    logger.add(
        log_file,
        level=level,
        rotation="10 MB",
        retention="14 days",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )

    _configure_stdlib_logging(level)

    logger.bind(module="script.run_scheduler").info(
        "Scheduler logging initialised at level {} file={}", level, log_file
    )
    console.log("[green]Scheduler logs[/] -> {}".format(log_file))


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
        _configure_logging(settings, override_level=args.log_level)
    except ValueError as exc:
        console.log("[bold red]Invalid log level[/] reason={}".format(exc))
        return 1

    try:
        from loreley.scheduler.main import main as scheduler_main
    except Exception as exc:  # pragma: no cover - defensive
        console.log("[bold red]Failed to import scheduler[/] reason={}".format(exc))
        logger.exception("Scheduler import failed")
        return 1

    # Delegate argument parsing and control flow to loreley.scheduler.main.
    try:
        return int(scheduler_main(forwarded))
    except Exception as exc:  # pragma: no cover - defensive
        console.log("[bold red]Scheduler failed to start[/] reason={}".format(exc))
        logger.exception("Scheduler crashed during startup")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())



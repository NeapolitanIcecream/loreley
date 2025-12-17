from __future__ import annotations

"""Entry script for running the Loreley read-only UI API (FastAPI).

Usage (with uv):

    uv run python script/run_api.py
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

from loguru import logger
from rich.console import Console

from loreley.config import Settings, get_settings

console = Console()
log = logger.bind(module="script.run_api")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Loreley read-only UI API (FastAPI).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument("--log-level", dest="log_level", help="Override Settings.log_level.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only).")
    return parser


class _LoguruInterceptHandler(logging.Handler):
    """Bridge standard-library logging records into Loguru."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(exception=record.exc_info).log(level, record.getMessage())


def _configure_stdlib_logging(level: str) -> None:
    handler: logging.Handler = _LoguruInterceptHandler()
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    logging.captureWarnings(True)


def _resolve_logs_dir(settings: Settings, role: str) -> Path:
    if settings.logs_base_dir:
        base_dir = Path(settings.logs_base_dir).expanduser()
    else:
        base_dir = Path.cwd()
    logs_root = base_dir / "logs"
    log_dir = logs_root / role
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _configure_logging(settings: Settings, *, override_level: str | None = None) -> None:
    level = (override_level or settings.log_level or "INFO").upper()
    try:
        logger.level(level)
    except ValueError as exc:
        raise ValueError(
            f"Invalid log level {level!r}; expected TRACE/DEBUG/INFO/SUCCESS/WARNING/ERROR/CRITICAL."
        ) from exc

    logger.remove()
    logger.add(sys.stderr, level=level, backtrace=False, diagnose=False)

    logs_dir = _resolve_logs_dir(settings, role="ui_api")
    log_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = logs_dir / f"ui-api-{log_timestamp}.log"
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

    log.info("UI API logging initialised level={} file={}", level, log_file)
    console.log("[green]UI API logs[/] -> {}".format(log_file))


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
        _configure_logging(settings, override_level=args.log_level)
    except ValueError as exc:
        console.log("[bold red]Invalid log level[/] reason={}".format(exc))
        return 1

    # Import uvicorn lazily so the base package can be installed without UI extras.
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover - dependency missing
        console.log(
            "[bold red]Missing UI dependencies[/] "
            "Install with `uv sync --extra ui` and retry. "
            f"reason={exc}"
        )
        return 1

    console.log(
        "[bold green]Loreley UI API online[/] "
        "host={} port={} db_host={!r}".format(args.host, args.port, settings.db_host),
    )
    log.info("Starting UI API host={} port={}", args.host, args.port)

    uvicorn.run(
        "loreley.api.app:app",
        host=str(args.host),
        port=int(args.port),
        reload=bool(args.reload),
        log_level=(args.log_level or settings.log_level or "info").lower(),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())



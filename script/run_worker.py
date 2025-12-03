from __future__ import annotations

"""Entry script for running the Loreley evolution worker.

This script:

- Loads application settings and configures Loguru logging, including routing
  standard-library logging (used by Dramatiq) through Loguru.
- Initialises the Dramatiq Redis broker defined in ``loreley.tasks.broker``.
- Imports ``loreley.tasks.workers`` so that the ``run_evolution_job`` actor is registered.
- Starts a single Dramatiq worker bound to the configured queue using a
  single-threaded worker pool.

Typical usage (with uv):

    uv run python script/run_worker.py
"""

import logging
import signal
import sys
import threading
from typing import Sequence

from dramatiq import Worker
from loguru import logger
from rich.console import Console

from loreley.config import get_settings
from loreley.tasks.broker import broker  # noqa: F401 - ensure broker is initialised
import loreley.tasks.workers as _workers  # noqa: F401  - register actors

console = Console()
log = logger.bind(module="script.run_worker")


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

    # Attach the intercept handler to the root logger so any library using the
    # standard logging module ends up in Loguru.
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # Ensure the Dramatiq logger is also captured explicitly.
    dramatiq_logger = logging.getLogger("dramatiq")
    dramatiq_logger.handlers = [handler]
    dramatiq_logger.setLevel(level)

    # Route warnings.warn() calls through the logging system as well.
    logging.captureWarnings(True)


def _configure_logging() -> None:
    """Configure Loguru and bridge stdlib logging using application settings."""

    settings = get_settings()
    level = (settings.log_level or "INFO").upper()

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        backtrace=False,
        diagnose=False,
    )

    _configure_stdlib_logging(level)

    log.info("Worker logging initialised at level {}", level)


def _install_signal_handlers(worker: Worker, stop_event: threading.Event | None = None) -> None:
    """Install SIGINT/SIGTERM handlers for graceful shutdown."""

    def _handle_signal(signum: int, _frame: object) -> None:
        console.log(
            f"[yellow]Received signal[/] signum={signum}; stopping worker...",
        )
        log.info("Worker received signal {}; stopping", signum)
        worker.stop()
        if stop_event is not None:
            stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    sigterm = getattr(signal, "SIGTERM", None)
    if sigterm is not None:
        signal.signal(sigterm, _handle_signal)


def main(_argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the evolution worker wrapper."""

    _configure_logging()
    settings = get_settings()

    console.log(
        "[bold green]Loreley worker online[/] "
        f"queue={settings.tasks_queue_name!r} worktree={settings.worker_repo_worktree!r}",
    )
    log.info(
        "Starting Loreley worker queue={} worktree={}",
        settings.tasks_queue_name,
        settings.worker_repo_worktree,
    )

    worker = Worker(broker, worker_threads=1)  # single-threaded worker
    stop_event = threading.Event()
    _install_signal_handlers(worker, stop_event=stop_event)

    try:
        worker.start()
        # Keep the main thread alive until a shutdown signal is received.
        stop_event.wait()
    except KeyboardInterrupt:
        console.log(
            "[yellow]Keyboard interrupt received[/]; shutting down worker...",
        )
        worker.stop()
    finally:
        # Ensure the worker is fully stopped before exiting.
        worker.stop()
        worker.join()
        console.log("[bold yellow]Loreley worker stopped[/]")
        log.info("Loreley worker stopped")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())



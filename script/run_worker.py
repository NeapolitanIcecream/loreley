from __future__ import annotations

"""Entry script for running the Loreley evolution worker.

This script:

- Loads application settings and configures Loguru logging.
- Initialises the Dramatiq Redis broker defined in ``app.tasks.broker``.
- Imports ``app.tasks.workers`` so that the ``run_evolution_job`` actor is registered.
- Starts a single Dramatiq worker bound to the configured queue using a
  single-threaded worker pool.

Typical usage (with uv):

    uv run python script/run_worker.py
"""

import signal
import sys
from typing import Sequence

from dramatiq import Worker
from loguru import logger
from rich.console import Console

from app.config import get_settings
from app.tasks.broker import broker  # noqa: F401 - ensure broker is initialised
import app.tasks.workers as _workers  # noqa: F401  - register actors

console = Console()
log = logger.bind(module="script.run_worker")


def _configure_logging() -> None:
    """Configure Loguru using application settings."""

    settings = get_settings()
    level = (settings.log_level or "INFO").upper()

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        backtrace=False,
        diagnose=False,
    )
    log.info("Worker logging initialised at level {}", level)


def _install_signal_handlers(worker: Worker) -> None:
    """Install SIGINT/SIGTERM handlers for graceful shutdown."""

    def _handle_signal(signum: int, _frame: object) -> None:
        console.log(
            f"[yellow]Received signal[/] signum={signum}; stopping worker...",
        )
        log.info("Worker received signal {}; stopping", signum)
        worker.stop()

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
    _install_signal_handlers(worker)

    try:
        worker.start()
        worker.join()
    except KeyboardInterrupt:
        console.log(
            "[yellow]Keyboard interrupt received[/]; shutting down worker...",
        )
        worker.stop()

    console.log("[bold yellow]Loreley worker stopped[/]")
    log.info("Loreley worker stopped")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())



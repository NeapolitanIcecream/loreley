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
import logging
import os
import signal
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Sequence

from dramatiq import Worker
from loguru import logger
from rich.console import Console

from loreley.config import Settings, get_settings

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

    # Attach the intercept handler to the root logger so any library using the
    # standard logging module ends up in Loguru.
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    dramatiq_logger = logging.getLogger("dramatiq")
    # Ensure Dramatiq logs propagate to the root logger and are processed once.
    dramatiq_logger.handlers.clear()
    dramatiq_logger.propagate = True
    dramatiq_logger.setLevel(level)

    # Route warnings.warn() calls through the logging system as well.
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

    logs_dir = _resolve_logs_dir(settings, role="worker")
    log_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = logs_dir / f"worker-{log_timestamp}.log"
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

    log.info("Worker logging initialised at level {} file={}", level, log_file)
    console.log("[green]Worker logs[/] -> {}".format(log_file))


def _install_signal_handlers(worker: Worker, stop_event: threading.Event | None = None) -> None:
    """Install SIGINT/SIGTERM handlers for graceful shutdown."""

    received_signals: dict[str, int] = {"count": 0}

    def _handle_signal(signum: int, _frame: object) -> None:
        received_signals["count"] += 1

        if received_signals["count"] == 1:
            console.log(
                f"[yellow]Received signal[/] signum={signum}; stopping worker...",
            )
            log.info("Worker received signal {}; stopping", signum)
            worker.stop()
            if stop_event is not None:
                stop_event.set()
            return

        console.log(
            f"[bold red]Second signal received[/] signum={signum}; forcing immediate shutdown.",
        )
        log.warning("Second signal {}; forcing immediate shutdown", signum)
        os._exit(130)

    signal.signal(signal.SIGINT, _handle_signal)
    sigterm = getattr(signal, "SIGTERM", None)
    if sigterm is not None:
        signal.signal(sigterm, _handle_signal)


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
        _configure_logging(settings, override_level=args.log_level)
    except ValueError as exc:
        console.log("[bold red]Invalid log level[/] reason={}".format(exc))
        return 1

    console.log(
        "[bold green]Loreley worker online[/] "
        f"queue={settings.tasks_queue_name!r} worktree={settings.worker_repo_worktree!r}",
    )
    log.info(
        "Starting Loreley worker queue={} worktree={}",
        settings.tasks_queue_name,
        settings.worker_repo_worktree,
    )

    try:
        # Lazily import the broker and worker actors after logging is configured
        # so that any configuration errors are surfaced cleanly to the user.
        from loreley.tasks import broker as broker_module
        import loreley.tasks.workers as _workers  # noqa: F401  - register actors

        dramatiq_broker = broker_module.broker
    except Exception as exc:  # pragma: no cover - defensive
        console.log(
            "[bold red]Failed to initialise worker dependencies[/] "
            f"reason={exc}. Check Redis/DB configuration and try again.",
        )
        log.exception("Worker bootstrap failed")
        return 1

    worker = Worker(dramatiq_broker, worker_threads=1)  # single-threaded worker
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
    except Exception as exc:  # pragma: no cover - defensive
        console.log("[bold red]Worker failed to start[/] reason={}".format(exc))
        log.exception("Worker crashed during startup")
        worker.stop()
        return 1
    finally:
        # Ensure the worker is fully stopped before exiting.
        worker.stop()
        worker.join()
        console.log("[bold yellow]Loreley worker stopped[/]")
        log.info("Loreley worker stopped")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())



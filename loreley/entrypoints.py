from __future__ import annotations

"""Process entrypoint helpers used by both the unified CLI and legacy scripts.

This module intentionally contains no business logic beyond:
- logging configuration
- preflight validation
- orchestration of long-running processes (scheduler / worker)
"""

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from dramatiq import Worker
from loguru import logger
from rich.console import Console

from loreley.config import Settings
from loreley.preflight import (
    has_failures,
    preflight_api,
    preflight_scheduler,
    preflight_ui,
    preflight_worker,
    render_results,
)

log = logger.bind(module="entrypoints")


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


def configure_process_logging(
    *,
    settings: Settings,
    console: Console,
    role: str,
    override_level: str | None = None,
) -> Path:
    """Configure Loguru sinks and bridge stdlib logging.

    Returns the log file path used for the rotating file sink.
    """
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

    logs_dir = _resolve_logs_dir(settings, role=role)
    log_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = logs_dir / f"{role}-{log_timestamp}.log"
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

    log.info("{} logging initialised level={} file={}", role, level, log_file)
    console.log(f"[green]{role} logs[/] -> {log_file}")
    return log_file


def _install_worker_signal_handlers(worker: Worker, *, console: Console, stop_event: threading.Event) -> None:
    """Install SIGINT/SIGTERM handlers for graceful shutdown."""
    received_signals: dict[str, int] = {"count": 0}

    def _handle_signal(signum: int, _frame: object) -> None:
        received_signals["count"] += 1
        if received_signals["count"] == 1:
            console.log(f"[yellow]Received signal[/] signum={signum}; stopping worker...")
            log.info("Worker received signal {}; stopping", signum)
            worker.stop()
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


def run_scheduler(
    *,
    settings: Settings,
    console: Console,
    argv: list[str] | None,
    preflight: bool = True,
    preflight_timeout_seconds: float = 2.0,
) -> int:
    """Run the Loreley scheduler (once or forever)."""
    if preflight:
        results = preflight_scheduler(settings, timeout_seconds=preflight_timeout_seconds)
        if has_failures(results):
            render_results(console, results, title="Loreley scheduler preflight")
            ok, warn, fail = (0, 0, 0)
            try:
                from loreley.preflight import summarize

                ok, warn, fail = summarize(results)
            except Exception:
                pass
            console.log(f"[bold red]Preflight failed[/] ok={ok} warn={warn} fail={fail}")
            console.log("Hint: start dependencies with `docker compose up -d postgres redis` and re-run.")
            return 1

    from loreley.scheduler.main import main as scheduler_main

    return int(scheduler_main(argv))


def run_worker(
    *,
    settings: Settings,
    console: Console,
    preflight: bool = True,
    preflight_timeout_seconds: float = 2.0,
) -> int:
    """Run the Loreley evolution worker as a single Dramatiq consumer process."""
    if preflight:
        results = preflight_worker(settings, timeout_seconds=preflight_timeout_seconds)
        if has_failures(results):
            render_results(console, results, title="Loreley worker preflight")
            ok, warn, fail = (0, 0, 0)
            try:
                from loreley.preflight import summarize

                ok, warn, fail = summarize(results)
            except Exception:
                pass
            console.log(f"[bold red]Preflight failed[/] ok={ok} warn={warn} fail={fail}")
            console.log("Hint: copy `env.example` to `.env`, fill required values, then retry.")
            return 1

    console.log(
        "[bold green]Loreley worker online[/] "
        f"queue={settings.tasks_queue_name!r} worktree={settings.worker_repo_worktree!r}",
    )

    try:
        # Lazily import the broker and worker actors after logging is configured so
        # that any configuration errors are surfaced cleanly to the user.
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
    _install_worker_signal_handlers(worker, console=console, stop_event=stop_event)

    try:
        worker.start()
        stop_event.wait()
    except KeyboardInterrupt:
        console.log("[yellow]Keyboard interrupt received[/]; shutting down worker...")
        worker.stop()
    except Exception as exc:  # pragma: no cover - defensive
        console.log("[bold red]Worker failed to start[/] reason={}".format(exc))
        log.exception("Worker crashed during startup")
        worker.stop()
        return 1
    finally:
        worker.stop()
        worker.join()
        console.log("[bold yellow]Loreley worker stopped[/]")
        log.info("Loreley worker stopped")
    return 0


def run_api(
    *,
    settings: Settings,
    console: Console,
    host: str,
    port: int,
    reload: bool,
    preflight: bool = True,
    preflight_timeout_seconds: float = 2.0,
    uvicorn_log_level: str | None = None,
) -> int:
    """Run the read-only UI API (FastAPI via uvicorn)."""
    if preflight:
        results = preflight_api(settings, timeout_seconds=preflight_timeout_seconds)
        if has_failures(results, treat_warnings_as_errors=True):
            render_results(console, results, title="Loreley UI API preflight")
            ok, warn, fail = (0, 0, 0)
            try:
                from loreley.preflight import summarize

                ok, warn, fail = summarize(results)
            except Exception:
                pass
            console.log(f"[bold red]Preflight failed[/] ok={ok} warn={warn} fail={fail}")
            console.log("Hint: install UI extras with `uv sync --extra ui` and ensure DB is reachable.")
            return 1

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
        "host={} port={} db_host={!r}".format(host, int(port), settings.db_host),
    )
    log.info("Starting UI API host={} port={}", host, int(port))

    uvicorn.run(
        "loreley.api.app:app",
        host=str(host),
        port=int(port),
        reload=bool(reload),
        log_level=(uvicorn_log_level or settings.log_level or "info").lower(),
    )
    return 0


_POLL_INTERVAL_SECONDS = 0.2
_STOP_TIMEOUT_SECONDS = 5.0


def _coerce_exit_code(returncode: int, *, stop_requested: bool) -> int:
    """Normalize subprocess return codes for a friendly CLI experience."""
    if stop_requested:
        return 0
    if returncode == 0:
        return 0
    if returncode < 0:
        signum = -returncode
        if signum in (signal.SIGINT, signal.SIGTERM):
            return 0
        return 128 + signum
    return int(returncode)


def _send_signal(proc: subprocess.Popen, signum: int) -> None:
    """Best-effort signal propagation to the Streamlit process (and its children)."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            # We start Streamlit in a new session so proc.pid is also the PGID.
            os.killpg(int(proc.pid), int(signum))
        else:  # pragma: no cover - Windows behavior
            proc.send_signal(signum)
    except ProcessLookupError:
        return
    except Exception:
        return


def _stop_proc(proc: subprocess.Popen, *, console: Console, first_signal: int) -> None:
    """Attempt a graceful shutdown, escalating to terminate/kill if needed."""
    _send_signal(proc, int(first_signal))
    try:
        proc.wait(timeout=_STOP_TIMEOUT_SECONDS)
        return
    except subprocess.TimeoutExpired:
        console.log("[yellow]UI did not stop after signal; terminating...[/]")

    try:
        if os.name == "posix":
            _send_signal(proc, signal.SIGTERM)
        else:  # pragma: no cover - Windows behavior
            proc.terminate()
    except Exception:
        pass

    try:
        proc.wait(timeout=_STOP_TIMEOUT_SECONDS)
        return
    except subprocess.TimeoutExpired:
        console.log("[bold red]UI did not terminate; killing...[/]")

    try:
        if os.name == "posix" and hasattr(signal, "SIGKILL"):
            _send_signal(proc, signal.SIGKILL)
        else:  # pragma: no cover - Windows behavior
            proc.kill()
    except Exception:
        pass

    try:
        proc.wait(timeout=_STOP_TIMEOUT_SECONDS)
    except Exception:
        pass


def run_ui(
    *,
    settings: Settings,
    console: Console,
    api_base_url: str,
    host: str,
    port: int,
    headless: bool,
    preflight: bool = True,
    preflight_timeout_seconds: float = 2.0,
) -> int:
    """Run the Streamlit UI.

    The UI is a read-only dashboard that calls the UI API via HTTP. It does not
    require direct database connectivity.
    """
    if preflight:
        results = preflight_ui(settings, timeout_seconds=preflight_timeout_seconds)
        if has_failures(results, treat_warnings_as_errors=True):
            render_results(console, results, title="Loreley UI preflight")
            ok, warn, fail = (0, 0, 0)
            try:
                from loreley.preflight import summarize

                ok, warn, fail = summarize(results)
            except Exception:
                pass
            console.log(f"[bold red]Preflight failed[/] ok={ok} warn={warn} fail={fail}")
            console.log("Hint: install UI extras with `uv sync --extra ui` and retry.")
            return 1

    env = dict(os.environ)
    env["LORELEY_UI_API_BASE_URL"] = str(api_base_url)

    ui_script = (Path(__file__).resolve().parent / "ui" / "app.py").resolve()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ui_script),
        "--server.address",
        str(host),
        "--server.port",
        str(int(port)),
    ]
    if headless:
        cmd += ["--server.headless", "true"]

    console.log(
        "[bold green]Loreley UI online[/] "
        "host={} port={} api_base_url={}".format(host, int(port), api_base_url),
    )

    stop_requested = False
    stop_signal = signal.SIGTERM
    proc: subprocess.Popen | None = None

    def _handle_sigterm(signum: int, _frame: object) -> None:
        nonlocal stop_requested, stop_signal, proc
        stop_requested = True
        stop_signal = int(signum)
        if proc is not None:
            _send_signal(proc, stop_signal)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        popen_kwargs: dict[str, object] = {"env": env}
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(cmd, **popen_kwargs)  # type: ignore[arg-type]
    except FileNotFoundError as exc:  # pragma: no cover
        console.log(
            "[bold red]Failed to start Streamlit[/] "
            "Install with `uv sync --extra ui` and retry. "
            f"reason={exc}"
        )
        return 1
    except KeyboardInterrupt:
        console.log("[yellow]Keyboard interrupt received[/]; exiting...")
        return 0

    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                return _coerce_exit_code(int(rc), stop_requested=stop_requested)

            if stop_requested:
                name = {signal.SIGTERM: "SIGTERM"}.get(stop_signal, str(stop_signal))
                console.log(f"[yellow]Stop signal received ({name})[/]; stopping UI...")
                _stop_proc(proc, console=console, first_signal=stop_signal)
                return 0

            time.sleep(_POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        console.log("[yellow]Keyboard interrupt received[/]; stopping UI...")
        _stop_proc(proc, console=console, first_signal=signal.SIGINT)
        return 0


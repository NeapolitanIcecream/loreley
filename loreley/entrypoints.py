from __future__ import annotations

"""Process entrypoint helpers used by the unified CLI and script wrappers.

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
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

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

    from loreley.naming import safe_namespace_from_settings

    exp_ns = safe_namespace_from_settings(settings)
    if not exp_ns:
        try:
            from loreley.db.base import INSTANCE_SCHEMA_VERSION, session_scope
            from loreley.db.instance import resolve_instance_namespace_from_marker

            with session_scope() as session:
                exp_ns = resolve_instance_namespace_from_marker(
                    session=session,
                    schema_version=INSTANCE_SCHEMA_VERSION,
                )
        except Exception:
            exp_ns = None
    logs_root = (base_dir / "logs" / exp_ns) if exp_ns else (base_dir / "logs")
    log_dir = logs_root / role
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _apply_dramatiq_prefetch_settings(*, settings: Settings, console: Console) -> None:
    """Apply configured Dramatiq prefetch limits before worker construction."""

    import dramatiq.worker as dramatiq_worker

    queue_prefetch = max(0, int(settings.tasks_queue_prefetch))
    delay_queue_prefetch = max(0, int(settings.tasks_delay_queue_prefetch))
    dramatiq_worker.QUEUE_PREFETCH = queue_prefetch
    dramatiq_worker.DELAY_QUEUE_PREFETCH = delay_queue_prefetch
    console.log(
        "[green]Dramatiq prefetch[/] queue={} delay={}".format(
            queue_prefetch,
            delay_queue_prefetch,
        )
    )
    log.info(
        "Applied Dramatiq prefetch settings queue_prefetch={} delay_queue_prefetch={}",
        queue_prefetch,
        delay_queue_prefetch,
    )


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


def reset_database(*, console: Console, yes: bool) -> int:
    """Drop and recreate all Loreley DB tables.

    This operation is destructive and should only be used for local/dev databases.
    """

    if not bool(yes):
        console.print("[bold red]Refusing to reset DB without --yes[/]")
        console.print("This will drop ALL tables and recreate them from ORM models.")
        return 2

    try:
        from loreley.db.base import reset_database_schema
        from loreley.tasks.broker import reset_redis_namespace

        reset_database_schema(include_console_log=True)
        deleted_keys = reset_redis_namespace()
        console.print(
            "[bold green]Redis namespace reset complete[/] deleted_keys={}".format(
                deleted_keys,
            )
        )
    except Exception as exc:  # pragma: no cover - defensive
        console.print(f"[bold red]Failed to reset database schema[/] reason={exc}")
        log.exception("Database schema reset failed: {}", exc)
        return 1

    return 0


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
    once: bool = False,
    auto_approve: bool = False,
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

    return int(
        scheduler_main(
            settings=settings,
            once=bool(once),
            auto_approve=bool(auto_approve),
        )
    )


def run_worker(
    *,
    settings: Settings,
    console: Console,
    preflight: bool = True,
    preflight_timeout_seconds: float = 2.0,
) -> int:
    """Run the Loreley evolution worker as a single Dramatiq consumer process."""
    attached_experiment = getattr(settings, "experiment_id", None)
    attached_raw = str(attached_experiment).strip() if attached_experiment is not None else ""

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

    if not attached_raw:
        console.log(
            "[bold red]Worker refused to start[/] "
            "reason=missing experiment attachment. "
            "Set EXPERIMENT_ID and retry.",
        )
        return 1

    from loreley.naming import resolve_experiment_identity, tasks_queue_name

    identity = resolve_experiment_identity(attached_raw)
    queue = tasks_queue_name(attached_raw)

    console.log(
        "[bold green]Loreley worker online[/] "
        f"experiment={identity.raw} experiment_uuid={identity.uuid} queue={queue!r} "
        f"worktree={settings.worker_repo_worktree!r}",
    )

    try:
        # Lazily import the broker and worker actors after logging is configured so
        # that any configuration errors are surfaced cleanly to the user.
        from loreley.tasks.broker import setup_broker
        from loreley.db.base import ensure_database_schema
        from loreley.tasks.workers import build_evolution_job_worker_actor

        dramatiq_broker = setup_broker(settings=settings)
        ensure_database_schema(settings=settings)
        # Register the experiment-attached actor bound to the derived queue.
        build_evolution_job_worker_actor(settings=settings, broker=dramatiq_broker)
    except Exception as exc:  # pragma: no cover - defensive
        console.log(
            "[bold red]Failed to initialise worker dependencies[/] "
            f"reason={exc}. Check Redis/DB configuration and try again.",
        )
        log.exception("Worker bootstrap failed")
        return 1

    _apply_dramatiq_prefetch_settings(settings=settings, console=console)
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
_API_STARTUP_MIN_WAIT_SECONDS = 10.0


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


def _api_health_url(api_base_url: str) -> str:
    """Return the health endpoint URL for a base URL."""
    return (api_base_url or "").rstrip("/") + "/api/v1/health"


def _is_ui_api_reachable(api_base_url: str, *, timeout_seconds: float) -> bool:
    """Return True when the UI API responds to GET /api/v1/health."""
    try:
        from loreley.net.http import HttpClient

        client = HttpClient(
            base_url=api_base_url,
            timeout_seconds=float(timeout_seconds),
            user_agent="loreley-ui",
        )
        return client.is_reachable("/api/v1/health", timeout_seconds=float(timeout_seconds))
    except Exception:
        return False


def _parse_local_http_base_url(api_base_url: str) -> tuple[str, int] | None:
    """Parse base URL and return (host, port) when auto-start is supported."""
    raw = (api_base_url or "").strip()
    if not raw:
        return None

    parsed = urlparse(raw)
    if parsed.scheme != "http":
        return None
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return None
    if parsed.port is None:
        return None
    path = (parsed.path or "").strip()
    if path and path != "/":
        return None
    return (str(parsed.hostname), int(parsed.port))


@dataclass
class _UiRunState:
    api_proc: subprocess.Popen | None = None
    owns_api_proc: bool = False
    stop_requested: bool = False
    stop_signal: int = signal.SIGTERM
    proc: subprocess.Popen | None = None


def _run_ui_preflight(
    *,
    settings: Settings,
    console: Console,
    preflight: bool,
    preflight_timeout_seconds: float,
) -> int | None:
    if not preflight:
        return None

    results = preflight_ui(settings, timeout_seconds=preflight_timeout_seconds)
    if not has_failures(results, treat_warnings_as_errors=True):
        return None

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


def _normalize_ui_api_base_url(api_base_url: str, *, console: Console) -> str | None:
    normalized = (api_base_url or "").strip()
    if normalized:
        return normalized
    console.log("[bold red]Invalid API base URL[/] value is empty.")
    return None


def _build_ui_environment(api_base_url: str) -> dict[str, str]:
    env = dict(os.environ)
    env["LORELEY_UI_API_BASE_URL"] = str(api_base_url)
    return env


def _popen_kwargs(env: dict[str, str]) -> dict[str, object]:
    popen_kwargs: dict[str, object] = {"env": env}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    return popen_kwargs


def _spawn_process(cmd: list[str], *, env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(cmd, **_popen_kwargs(env))  # type: ignore[arg-type]


def _build_ui_api_command(
    *,
    settings: Settings,
    api_host: str,
    api_port: int,
    preflight: bool,
    preflight_timeout_seconds: float,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "loreley",
        "--log-level",
        str(settings.log_level or "INFO"),
        "api",
        "--host",
        str(api_host),
        "--port",
        str(int(api_port)),
        "--preflight-timeout-seconds",
        str(float(preflight_timeout_seconds)),
    ]
    if not preflight:
        cmd.append("--no-preflight")
    return cmd


def _build_streamlit_command(*, host: str, port: int, headless: bool) -> list[str]:
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
    return cmd


def _install_ui_signal_handler(state: _UiRunState) -> None:
    def _handle_sigterm(signum: int, _frame: object) -> None:
        state.stop_requested = True
        state.stop_signal = int(signum)
        if state.proc is not None:
            _send_signal(state.proc, state.stop_signal)
        if state.owns_api_proc and state.api_proc is not None:
            _send_signal(state.api_proc, state.stop_signal)

    signal.signal(signal.SIGTERM, _handle_sigterm)


def _stop_owned_ui_api(
    state: _UiRunState,
    *,
    console: Console,
    first_signal: int,
) -> None:
    if state.owns_api_proc and state.api_proc is not None:
        _stop_proc(state.api_proc, console=console, first_signal=first_signal)


def _wait_for_ui_api_startup(
    *,
    state: _UiRunState,
    console: Console,
    api_base_url: str,
    preflight_timeout_seconds: float,
) -> int | None:
    api_proc = state.api_proc
    if api_proc is None:
        return None

    startup_wait = max(_API_STARTUP_MIN_WAIT_SECONDS, float(preflight_timeout_seconds) * 10.0)
    deadline = time.time() + startup_wait
    with console.status(f"[bold]Waiting for UI API[/] url={_api_health_url(api_base_url)}"):
        while time.time() < deadline:
            if state.stop_requested:
                console.log("[yellow]Stop signal received[/]; stopping UI API...")
                _stop_proc(api_proc, console=console, first_signal=state.stop_signal)
                return 0
            if api_proc.poll() is not None:
                rc = int(api_proc.returncode or 0)
                console.log(f"[bold red]UI API exited during startup[/] rc={rc}")
                return _coerce_exit_code(rc, stop_requested=False)
            if _is_ui_api_reachable(api_base_url, timeout_seconds=preflight_timeout_seconds):
                console.log("[bold green]UI API ready[/]")
                return None
            time.sleep(_POLL_INTERVAL_SECONDS)

    console.log(
        "[bold red]Timed out waiting for UI API[/] "
        f"url={_api_health_url(api_base_url)!r} timeout_seconds={startup_wait}",
    )
    _stop_proc(api_proc, console=console, first_signal=signal.SIGTERM)
    return 1


def _maybe_start_local_ui_api(
    *,
    state: _UiRunState,
    settings: Settings,
    console: Console,
    api_base_url: str,
    env: dict[str, str],
    preflight: bool,
    preflight_timeout_seconds: float,
) -> int | None:
    if _is_ui_api_reachable(api_base_url, timeout_seconds=preflight_timeout_seconds):
        return None

    local_target = _parse_local_http_base_url(api_base_url)
    if local_target is None:
        console.log(
            "[yellow]UI API not reachable[/] "
            f"url={_api_health_url(api_base_url)!r} (auto-start disabled for non-local or non-http base URLs)",
        )
        return None

    api_host, api_port = local_target
    # Re-check before spawning to avoid races when another process starts the API.
    if _is_ui_api_reachable(api_base_url, timeout_seconds=preflight_timeout_seconds):
        return None

    console.log(
        "[yellow]UI API not reachable[/] "
        f"starting automatically host={api_host} port={int(api_port)}",
    )
    api_cmd = _build_ui_api_command(
        settings=settings,
        api_host=api_host,
        api_port=api_port,
        preflight=preflight,
        preflight_timeout_seconds=preflight_timeout_seconds,
    )

    try:
        state.api_proc = _spawn_process(api_cmd, env=env)
        state.owns_api_proc = True
    except Exception as exc:  # pragma: no cover - defensive
        console.log(f"[bold red]Failed to start UI API automatically[/] reason={exc}")
        log.exception("Failed to spawn UI API process")
        return 1

    return _wait_for_ui_api_startup(
        state=state,
        console=console,
        api_base_url=api_base_url,
        preflight_timeout_seconds=preflight_timeout_seconds,
    )


def _start_streamlit_process(
    *,
    state: _UiRunState,
    console: Console,
    env: dict[str, str],
    api_base_url: str,
    host: str,
    port: int,
    headless: bool,
) -> int | None:
    cmd = _build_streamlit_command(host=host, port=port, headless=headless)
    console.log(
        "[bold green]Loreley UI online[/] "
        "host={} port={} api_base_url={}".format(host, int(port), api_base_url),
    )

    try:
        state.proc = _spawn_process(cmd, env=env)
    except FileNotFoundError as exc:  # pragma: no cover
        console.log(
            "[bold red]Failed to start Streamlit[/] "
            "Install with `uv sync --extra ui` and retry. "
            f"reason={exc}"
        )
        _stop_owned_ui_api(state, console=console, first_signal=signal.SIGTERM)
        return 1
    except KeyboardInterrupt:
        console.log("[yellow]Keyboard interrupt received[/]; exiting...")
        _stop_owned_ui_api(state, console=console, first_signal=signal.SIGINT)
        return 0
    return None


def _poll_ui_processes_once(*, state: _UiRunState, console: Console) -> int | None:
    proc = state.proc
    if proc is None:
        return 1

    rc = proc.poll()
    if rc is not None:
        _stop_owned_ui_api(state, console=console, first_signal=signal.SIGTERM)
        return _coerce_exit_code(int(rc), stop_requested=state.stop_requested)

    if state.owns_api_proc and state.api_proc is not None and state.api_proc.poll() is not None:
        api_rc = int(state.api_proc.returncode or 0)
        console.log(f"[bold red]UI API process exited[/] rc={api_rc}; stopping UI...")
        _stop_proc(proc, console=console, first_signal=signal.SIGTERM)
        return _coerce_exit_code(api_rc, stop_requested=False)

    if state.stop_requested:
        name = {signal.SIGTERM: "SIGTERM"}.get(state.stop_signal, str(state.stop_signal))
        console.log(f"[yellow]Stop signal received ({name})[/]; stopping UI...")
        _stop_proc(proc, console=console, first_signal=state.stop_signal)
        _stop_owned_ui_api(state, console=console, first_signal=state.stop_signal)
        return 0

    return None


def _monitor_ui_processes(*, state: _UiRunState, console: Console) -> int:
    proc = state.proc
    if proc is None:
        return 1

    try:
        while True:
            result = _poll_ui_processes_once(state=state, console=console)
            if result is not None:
                return result
            time.sleep(_POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        console.log("[yellow]Keyboard interrupt received[/]; stopping UI...")
        _stop_proc(proc, console=console, first_signal=signal.SIGINT)
        _stop_owned_ui_api(state, console=console, first_signal=signal.SIGINT)
        return 0


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
    preflight_rc = _run_ui_preflight(
        settings=settings,
        console=console,
        preflight=preflight,
        preflight_timeout_seconds=preflight_timeout_seconds,
    )
    if preflight_rc is not None:
        return preflight_rc

    normalized_api_base_url = _normalize_ui_api_base_url(api_base_url, console=console)
    if normalized_api_base_url is None:
        return 1

    env = _build_ui_environment(normalized_api_base_url)
    state = _UiRunState()
    _install_ui_signal_handler(state)

    try:
        api_start_rc = _maybe_start_local_ui_api(
            state=state,
            settings=settings,
            console=console,
            api_base_url=normalized_api_base_url,
            env=env,
            preflight=preflight,
            preflight_timeout_seconds=preflight_timeout_seconds,
        )
        if api_start_rc is not None:
            return api_start_rc
    except KeyboardInterrupt:
        console.log("[yellow]Keyboard interrupt received[/]; stopping...")
        _stop_owned_ui_api(state, console=console, first_signal=signal.SIGINT)
        return 0

    streamlit_start_rc = _start_streamlit_process(
        state=state,
        console=console,
        env=env,
        api_base_url=normalized_api_base_url,
        host=host,
        port=port,
        headless=headless,
    )
    if streamlit_start_rc is not None:
        return streamlit_start_rc

    return _monitor_ui_processes(state=state, console=console)

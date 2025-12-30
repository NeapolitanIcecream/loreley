from __future__ import annotations

"""Unified CLI for Loreley.

This CLI is designed to:
- provide a single entrypoint (`loreley ...`)
- run preflight checks before starting long-running processes
- keep legacy `script/run_*.py` wrappers usable for local development
"""

import argparse
import os
import sys
from typing import Sequence

from rich.console import Console

from loreley.config import get_settings
from loreley.entrypoints import configure_process_logging, run_api, run_scheduler, run_ui, run_worker
from loreley.preflight import (
    CheckResult,
    has_failures,
    preflight_all,
    preflight_api,
    preflight_scheduler,
    preflight_ui,
    preflight_worker,
    render_results,
    summarize,
    to_json,
)

console = Console()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loreley",
        description="Loreley unified CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        help="Override LOG_LEVEL for this invocation (TRACE/DEBUG/INFO/WARNING/ERROR).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Run environment preflight checks.")
    doctor.add_argument(
        "--role",
        default="all",
        choices=("all", "scheduler", "worker", "api", "ui"),
        help="Which component you want to validate.",
    )
    doctor.add_argument(
        "--timeout-seconds",
        type=float,
        default=2.0,
        help="Network timeout used for DB/Redis connectivity checks.",
    )
    doctor.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures (non-zero exit code).",
    )
    doctor.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print results as JSON (useful for CI).",
    )

    scheduler = subparsers.add_parser("scheduler", help="Run the evolution scheduler.")
    scheduler.add_argument(
        "--once",
        action="store_true",
        help="Execute a single scheduling tick and exit.",
    )
    scheduler.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip preflight validation.",
    )
    scheduler.add_argument(
        "--preflight-timeout-seconds",
        type=float,
        default=2.0,
        help="Network timeout used for DB/Redis connectivity checks.",
    )

    worker = subparsers.add_parser("worker", help="Run the evolution worker (Dramatiq consumer).")
    worker.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip preflight validation.",
    )
    worker.add_argument(
        "--preflight-timeout-seconds",
        type=float,
        default=2.0,
        help="Network timeout used for DB/Redis connectivity checks.",
    )

    api = subparsers.add_parser("api", help="Run the read-only UI API (FastAPI via uvicorn).")
    api.add_argument("--host", default="127.0.0.1", help="Bind host.")
    api.add_argument("--port", type=int, default=8000, help="Bind port.")
    api.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only).")
    api.add_argument("--no-preflight", action="store_true", help="Skip preflight validation.")
    api.add_argument(
        "--preflight-timeout-seconds",
        type=float,
        default=2.0,
        help="Network timeout used for DB connectivity checks.",
    )

    ui = subparsers.add_parser("ui", help="Run the Streamlit UI.")
    ui.add_argument(
        "--api-base-url",
        default=os.getenv("LORELEY_UI_API_BASE_URL", "http://127.0.0.1:8000"),
        help="Base URL of the Loreley UI API.",
    )
    ui.add_argument("--host", default="127.0.0.1", help="Streamlit bind host.")
    ui.add_argument("--port", type=int, default=8501, help="Streamlit bind port.")
    ui.add_argument("--headless", action="store_true", help="Run without opening a browser.")
    ui.add_argument("--no-preflight", action="store_true", help="Skip preflight validation.")
    ui.add_argument(
        "--preflight-timeout-seconds",
        type=float,
        default=2.0,
        help="Network timeout used for preflight checks.",
    )

    return parser


def _run_doctor(*, role: str, timeout_seconds: float, strict: bool, json_output: bool) -> int:
    settings = get_settings()
    timeout = float(max(0.2, timeout_seconds))

    results: list[CheckResult]
    if role == "scheduler":
        results = preflight_scheduler(settings, timeout_seconds=timeout)
    elif role == "worker":
        results = preflight_worker(settings, timeout_seconds=timeout)
    elif role == "api":
        results = preflight_api(settings, timeout_seconds=timeout)
    elif role == "ui":
        results = preflight_ui(settings, timeout_seconds=timeout)
    else:
        results = preflight_all(settings, timeout_seconds=timeout)

    if json_output:
        console.print(to_json(results))
    else:
        render_results(console, results, title="Loreley doctor")

    ok, warn, fail = summarize(results)
    if fail:
        console.print(f"[bold red]Doctor failed[/] ok={ok} warn={warn} fail={fail}")
        return 1
    if warn and strict:
        console.print(f"[bold yellow]Doctor warnings (strict)[/] ok={ok} warn={warn} fail={fail}")
        return 2
    console.print(f"[bold green]Doctor passed[/] ok={ok} warn={warn} fail={fail}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Console script entrypoint."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Load settings after parsing so that `--help` works without a configured environment.
    try:
        settings = get_settings()
    except Exception as exc:  # pragma: no cover - defensive
        console.print(
            "[bold red]Invalid Loreley configuration[/] "
            f"reason={exc}. Copy `env.example` to `.env` and set required values.",
        )
        return 1

    role = str(args.command)
    log_role = {"api": "ui_api"}.get(role, role)
    try:
        configure_process_logging(
            settings=settings,
            console=console,
            role=log_role,
            override_level=getattr(args, "log_level", None),
        )
    except ValueError as exc:
        console.print(f"[bold red]Invalid log level[/] reason={exc}")
        return 1

    if args.command == "doctor":
        return _run_doctor(
            role=str(args.role),
            timeout_seconds=float(args.timeout_seconds),
            strict=bool(args.strict),
            json_output=bool(args.json_output),
        )

    if args.command == "scheduler":
        forwarded: list[str] = ["--once"] if bool(args.once) else []
        return run_scheduler(
            settings=settings,
            console=console,
            argv=forwarded,
            preflight=not bool(args.no_preflight),
            preflight_timeout_seconds=float(args.preflight_timeout_seconds),
        )

    if args.command == "worker":
        return run_worker(
            settings=settings,
            console=console,
            preflight=not bool(args.no_preflight),
            preflight_timeout_seconds=float(args.preflight_timeout_seconds),
        )

    if args.command == "api":
        return run_api(
            settings=settings,
            console=console,
            host=str(args.host),
            port=int(args.port),
            reload=bool(args.reload),
            preflight=not bool(args.no_preflight),
            preflight_timeout_seconds=float(args.preflight_timeout_seconds),
            uvicorn_log_level=getattr(args, "log_level", None),
        )

    if args.command == "ui":
        return run_ui(
            settings=settings,
            console=console,
            api_base_url=str(args.api_base_url),
            host=str(args.host),
            port=int(args.port),
            headless=bool(args.headless),
            preflight=not bool(args.no_preflight),
            preflight_timeout_seconds=float(args.preflight_timeout_seconds),
        )

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))



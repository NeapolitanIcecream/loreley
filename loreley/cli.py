from __future__ import annotations

"""Unified CLI for Loreley.

This CLI is designed to:
- provide a single entrypoint (`loreley ...`)
- run preflight checks before starting long-running processes
"""

import os
import sys
import json
from enum import Enum
from typing import Sequence

import click
import typer
from rich.console import Console

from loreley.config import Settings, get_settings
from loreley.entrypoints import configure_process_logging, reset_database, run_api, run_scheduler, run_ui, run_worker
from loreley.preflight import (
    CheckResult,
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
app = typer.Typer(add_completion=False, help="Loreley unified CLI.")
config_app = typer.Typer(help="Inspect effective Loreley configuration.")
app.add_typer(config_app, name="config")
archive_app = typer.Typer(help="Inspect MAP-Elites archives.")
app.add_typer(archive_app, name="archive")


class DoctorRole(str, Enum):
    all = "all"
    scheduler = "scheduler"
    worker = "worker"
    api = "api"
    ui = "ui"


def _load_settings_or_exit() -> Settings:
    try:
        return get_settings()
    except Exception as exc:  # pragma: no cover - defensive
        console.print(
            "[bold red]Invalid Loreley configuration[/] "
            f"reason={exc}. Copy `env.example` to `.env` and set required values.",
        )
        raise typer.Exit(code=1) from exc


def _configure_logging_or_exit(*, settings: Settings, role: str, override_level: str | None) -> None:
    try:
        configure_process_logging(
            settings=settings,
            console=console,
            role=role,
            override_level=override_level,
        )
    except ValueError as exc:
        console.print(f"[bold red]Invalid log level[/] reason={exc}")
        raise typer.Exit(code=1) from exc


def _run_doctor(
    *,
    settings: Settings,
    role: str,
    timeout_seconds: float,
    strict: bool,
    json_output: bool,
) -> int:
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


@app.callback()
def _callback(
    ctx: typer.Context,
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        help="Override LOG_LEVEL for this invocation (TRACE/DEBUG/INFO/WARNING/ERROR).",
    ),
) -> None:
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level


def _get_log_level(ctx: typer.Context) -> str | None:
    obj = getattr(ctx, "obj", None) or {}
    level = obj.get("log_level")
    return str(level) if level else None


@config_app.command("dump")
def config_dump(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print effective settings as JSON.",
        show_default=True,
    ),
    yaml_output: bool = typer.Option(
        False,
        "--yaml",
        help="Print effective settings as YAML.",
        show_default=True,
    ),
    mask_secrets: bool = typer.Option(
        True,
        "--mask-secrets/--no-mask-secrets",
        help="Mask credentials and API keys in the output.",
        show_default=True,
    ),
) -> None:
    """Dump effective configuration for reproducibility and troubleshooting."""
    if json_output and yaml_output:
        typer.echo(
            "Invalid output format: choose exactly one output format via --json or --yaml.",
        )
        raise typer.Exit(code=1)

    settings = _load_settings_or_exit()
    payload = settings.export_safe(mask_secrets=bool(mask_secrets))

    if yaml_output:
        try:
            import yaml
        except Exception as exc:  # pragma: no cover - defensive
            typer.echo("YAML output is unavailable: install PyYAML first.")
            raise typer.Exit(code=1) from exc
        serialized = yaml.safe_dump(payload, allow_unicode=True, sort_keys=True)
    else:
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)

    typer.echo(serialized)


@app.command()
def doctor(
    ctx: typer.Context,
    role: DoctorRole = typer.Option(
        DoctorRole.all,
        "--role",
        help="Which component you want to validate.",
        show_default=True,
    ),
    timeout_seconds: float = typer.Option(
        2.0,
        "--timeout-seconds",
        help="Network timeout used for DB/Redis connectivity checks.",
        show_default=True,
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Treat warnings as failures (non-zero exit code).",
        show_default=True,
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print results as JSON (useful for CI).",
        show_default=True,
    ),
) -> None:
    """Run environment preflight checks."""
    settings = _load_settings_or_exit()
    _configure_logging_or_exit(settings=settings, role="doctor", override_level=_get_log_level(ctx))
    code = _run_doctor(
        settings=settings,
        role=str(role.value),
        timeout_seconds=float(timeout_seconds),
        strict=bool(strict),
        json_output=bool(json_output),
    )
    raise typer.Exit(code=int(code))


@app.command()
def scheduler(
    ctx: typer.Context,
    once: bool = typer.Option(False, "--once", help="Execute a single scheduling tick and exit."),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Auto-approve startup approval and start without prompting (useful for CI/containers).",
    ),
    no_preflight: bool = typer.Option(False, "--no-preflight", help="Skip preflight validation."),
    preflight_timeout_seconds: float = typer.Option(
        2.0,
        "--preflight-timeout-seconds",
        help="Network timeout used for DB/Redis connectivity checks.",
        show_default=True,
    ),
) -> None:
    """Run the evolution scheduler."""
    settings = _load_settings_or_exit()
    _configure_logging_or_exit(settings=settings, role="scheduler", override_level=_get_log_level(ctx))
    code = run_scheduler(
        settings=settings,
        console=console,
        once=bool(once),
        auto_approve=bool(yes),
        preflight=not bool(no_preflight),
        preflight_timeout_seconds=float(preflight_timeout_seconds),
    )
    raise typer.Exit(code=int(code))


@app.command()
def worker(
    ctx: typer.Context,
    no_preflight: bool = typer.Option(False, "--no-preflight", help="Skip preflight validation."),
    preflight_timeout_seconds: float = typer.Option(
        2.0,
        "--preflight-timeout-seconds",
        help="Network timeout used for DB/Redis connectivity checks.",
        show_default=True,
    ),
) -> None:
    """Run the evolution worker (Dramatiq consumer)."""
    settings = _load_settings_or_exit()
    _configure_logging_or_exit(settings=settings, role="worker", override_level=_get_log_level(ctx))
    code = run_worker(
        settings=settings,
        console=console,
        preflight=not bool(no_preflight),
        preflight_timeout_seconds=float(preflight_timeout_seconds),
    )
    raise typer.Exit(code=int(code))


@app.command()
def api(
    ctx: typer.Context,
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host.", show_default=True),
    port: int = typer.Option(8000, "--port", help="Bind port.", show_default=True),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev only)."),
    no_preflight: bool = typer.Option(False, "--no-preflight", help="Skip preflight validation."),
    preflight_timeout_seconds: float = typer.Option(
        2.0,
        "--preflight-timeout-seconds",
        help="Network timeout used for DB connectivity checks.",
        show_default=True,
    ),
) -> None:
    """Run the read-only UI API (FastAPI via uvicorn)."""
    settings = _load_settings_or_exit()
    log_level = _get_log_level(ctx)
    _configure_logging_or_exit(settings=settings, role="ui_api", override_level=log_level)
    code = run_api(
        settings=settings,
        console=console,
        host=str(host),
        port=int(port),
        reload=bool(reload),
        preflight=not bool(no_preflight),
        preflight_timeout_seconds=float(preflight_timeout_seconds),
        uvicorn_log_level=log_level,
    )
    raise typer.Exit(code=int(code))


@app.command()
def ui(
    ctx: typer.Context,
    api_base_url: str | None = typer.Option(
        None,
        "--api-base-url",
        help="Base URL of the Loreley UI API.",
        show_default=False,
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Streamlit bind host.", show_default=True),
    port: int = typer.Option(8501, "--port", help="Streamlit bind port.", show_default=True),
    headless: bool = typer.Option(False, "--headless", help="Run without opening a browser."),
    no_preflight: bool = typer.Option(False, "--no-preflight", help="Skip preflight validation."),
    preflight_timeout_seconds: float = typer.Option(
        2.0,
        "--preflight-timeout-seconds",
        help="Network timeout used for preflight checks.",
        show_default=True,
    ),
) -> None:
    """Run the Streamlit UI."""
    settings = _load_settings_or_exit()
    _configure_logging_or_exit(settings=settings, role="ui", override_level=_get_log_level(ctx))

    api_base_url = (api_base_url or "").strip() or os.getenv("LORELEY_UI_API_BASE_URL", "http://127.0.0.1:8000")
    code = run_ui(
        settings=settings,
        console=console,
        api_base_url=str(api_base_url),
        host=str(host),
        port=int(port),
        headless=bool(headless),
        preflight=not bool(no_preflight),
        preflight_timeout_seconds=float(preflight_timeout_seconds),
    )
    raise typer.Exit(code=int(code))


@app.command("reset-db")
def reset_db(
    ctx: typer.Context,
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Confirm that you want to irreversibly drop all tables.",
        show_default=True,
    ),
) -> None:
    """Drop and recreate all Loreley DB tables."""
    settings = _load_settings_or_exit()
    _configure_logging_or_exit(settings=settings, role="db", override_level=_get_log_level(ctx))
    code = reset_database(console=console, yes=bool(yes))
    raise typer.Exit(code=int(code))


@archive_app.command("stats")
def archive_stats(
    ctx: typer.Context,
    island_id: str | None = typer.Option(
        None,
        "--island-id",
        help="Island ID; empty means the default island.",
        show_default=False,
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print stats as JSON.",
        show_default=True,
    ),
) -> None:
    """Print MAP-Elites archive stats for an island."""
    settings = _load_settings_or_exit()
    _configure_logging_or_exit(settings=settings, role="archive", override_level=_get_log_level(ctx))
    effective_island = (island_id or "").strip() or (settings.mapelites_default_island_id or "main")

    try:
        from loreley.core.map_elites.map_elites import MapElitesManager

        manager = MapElitesManager(settings=settings)
        stats = dict(manager.describe_island(effective_island))
    except Exception as exc:  # pragma: no cover - defensive
        console.print(
            "[bold red]Failed to load archive stats[/] "
            f"island={effective_island} reason={exc}",
        )
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
        return

    from rich.table import Table

    table = Table(title="MAP-Elites archive stats")
    table.add_column("field", style="bold")
    table.add_column("value")
    for key in (
        "island_id",
        "occupied",
        "cells",
        "coverage",
        "qd_score",
        "norm_qd_score",
        "best_fitness",
    ):
        if key not in stats:
            continue
        table.add_row(str(key), str(stats.get(key)))
    console.print(table)


def main(argv: Sequence[str] | None = None) -> int:
    """Console script entrypoint."""
    args = list(argv) if argv is not None else None
    try:
        result = app(prog_name="loreley", args=args, standalone_mode=False)
        if isinstance(result, int):
            return int(result)
        return 0
    except click.ClickException as exc:
        exc.show()
        return int(getattr(exc, "exit_code", 1) or 1)
    except click.Abort:
        console.print("[yellow]Aborted[/]")
        return 1
    except typer.Exit as exc:
        exit_code = getattr(exc, "exit_code", None)
        if exit_code is None:
            exit_code = getattr(exc, "code", 0)
        return int(exit_code or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))



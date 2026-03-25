from __future__ import annotations

"""Unified CLI for Loreley.

This CLI is designed to:
- provide a single entrypoint (`loreley ...`)
- run preflight checks before starting long-running processes
"""

from datetime import datetime, timezone
import os
import sys
import json
from enum import Enum
from typing import Any, Sequence
import uuid

import click
import typer
from rich.console import Console

from loreley.config import Settings, get_settings, resolve_default_island_id
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
jobs_app = typer.Typer(help="Inspect and repair evolution jobs.")
app.add_typer(config_app, name="config")
app.add_typer(jobs_app, name="jobs")
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


def _resolve_effective_island(*, settings: Settings, island_id: str | None) -> str:
    raw = (island_id or "").strip()
    if raw:
        return raw
    return resolve_default_island_id(settings)


def _load_archive_stats_or_exit(*, settings: Settings, island_id: str) -> dict[str, object]:
    try:
        from loreley.core.map_elites.map_elites import MapElitesManager

        manager = MapElitesManager(settings=settings)
        return dict(manager.describe_island(island_id))
    except Exception as exc:  # pragma: no cover - defensive
        console.print(
            "[bold red]Failed to load archive stats[/] "
            f"island={island_id} reason={exc}",
        )
        raise typer.Exit(code=1) from exc


def _iso_or_none(value: object) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return None


def _display_or_na(value: object) -> str:
    if value is None:
        return "n/a"
    text = str(value)
    return text if text else "n/a"


def _job_status_value(value: object) -> str:
    status_value = getattr(value, "value", value)
    return str(status_value or "").strip().lower()


def _job_lease_payload(*, job: Any, now: datetime) -> dict[str, object]:
    lease_expires_at = getattr(job, "lease_expires_at", None)
    run_token = getattr(job, "run_token", None)
    worker_id = getattr(job, "worker_id", None)
    status = _job_status_value(getattr(job, "status", None))

    if status == "running":
        if run_token is None or worker_id is None or lease_expires_at is None:
            state = "missing"
        elif isinstance(lease_expires_at, datetime) and lease_expires_at < now:
            state = "stale"
        else:
            state = "active"
    else:
        state = "none"

    return {
        "state": state,
        "heartbeat_at": _iso_or_none(getattr(job, "heartbeat_at", None)),
        "lease_expires_at": _iso_or_none(lease_expires_at),
        "run_token": str(run_token) if run_token is not None else None,
        "worker_id": str(worker_id) if worker_id is not None else None,
    }


def _job_summary_payload(*, job: Any, now: datetime) -> dict[str, object]:
    return {
        "job_id": str(getattr(job, "id", "")),
        "status": _job_status_value(getattr(job, "status", None)),
        "base_commit_hash": getattr(job, "base_commit_hash", None),
        "island_id": getattr(job, "island_id", None),
        "recovery_count": int(getattr(job, "recovery_count", 0) or 0),
        "result_commit_hash": getattr(job, "result_commit_hash", None),
        "last_error": getattr(job, "last_error", None),
        "created_at": _iso_or_none(getattr(job, "created_at", None)),
        "completed_at": _iso_or_none(getattr(job, "completed_at", None)),
        "lease_state": str(_job_lease_payload(job=job, now=now)["state"]),
    }


def _failed_stale_job_conditions(
    *,
    EvolutionJob: Any,
    JobStatus: Any,
    func: Any,
    max_recovery_attempts: int,
) -> tuple[Any, ...]:
    lease_error_norm = func.lower(func.trim(func.coalesce(EvolutionJob.last_error, "")))
    return (
        EvolutionJob.status == JobStatus.FAILED,
        EvolutionJob.recovery_count > int(max_recovery_attempts),
        lease_error_norm.like("lease expired after missing heartbeat;%"),
    )


def _job_retry_state(*, job: Any, now: datetime) -> tuple[bool, str | None]:
    from loreley.db.models import JobStatus

    status = getattr(job, "status", None)
    if status == JobStatus.FAILED:
        return True, None

    lease_state = str(_job_lease_payload(job=job, now=now)["state"])
    if status == JobStatus.RUNNING and lease_state in {"missing", "stale"}:
        return True, lease_state
    return False, lease_state


def _retry_job_row(*, job: Any, reason: str, now: datetime) -> dict[str, object]:
    previous_status = _job_status_value(getattr(job, "status", None))
    previous_recovery_count = int(getattr(job, "recovery_count", 0) or 0)
    from loreley.db.models import JobStatus

    job.status = JobStatus.PENDING
    job.scheduled_at = now
    job.started_at = None
    job.completed_at = None
    job.heartbeat_at = None
    job.lease_expires_at = None
    job.run_token = None
    job.worker_id = None
    job.recovery_count = 0
    job.result_commit_hash = None
    job.last_error = str(reason or "").strip() or "manual retry requested via CLI"
    return {
        "job_id": str(getattr(job, "id", "")),
        "previous_status": previous_status,
        "new_status": _job_status_value(getattr(job, "status", None)),
        "recovery_count_reset_from": previous_recovery_count,
        "reason": job.last_error,
    }


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


@app.command()
def status(
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
        help="Print status as JSON.",
        show_default=True,
    ),
) -> None:
    """Print a high-level operational status summary."""
    settings = _load_settings_or_exit()
    _configure_logging_or_exit(settings=settings, role="status", override_level=_get_log_level(ctx))
    effective_island = _resolve_effective_island(settings=settings, island_id=island_id)

    def _short_hash(value: str | None) -> str:
        raw = str(value or "").strip()
        if not raw:
            return "n/a"
        return raw[:12] if len(raw) > 12 else raw

    try:
        from sqlalchemy import func, or_, select

        from loreley.db.base import session_scope
        from loreley.db.models import CommitCard, EvolutionJob, InstanceMetadata, JobStatus, Metric

        with session_scope() as session:
            instance = session.get(InstanceMetadata, 1)
            if instance is None:
                from loreley.db.instance import INIT_DB_HINT

                console.print(f"[bold red]Instance metadata is missing[/] {INIT_DB_HINT}")
                raise typer.Exit(code=1)

            unfinished_statuses = (
                JobStatus.PENDING,
                JobStatus.QUEUED,
                JobStatus.RUNNING,
            )
            stmt_unfinished = select(func.count(EvolutionJob.id)).where(
                EvolutionJob.status.in_(unfinished_statuses),
            )
            unfinished_jobs = int(session.execute(stmt_unfinished).scalar_one())

            status_norm = func.lower(func.trim(func.coalesce(EvolutionJob.ingestion_status, "")))
            commit_norm = func.trim(func.coalesce(EvolutionJob.result_commit_hash, ""))
            stmt_pending_ingest = (
                select(func.count(EvolutionJob.id))
                .where(EvolutionJob.status == JobStatus.SUCCEEDED)
                .where(status_norm.not_in(("succeeded", "skipped")))
                .where(commit_norm != "")
            )
            pending_ingestion_jobs = int(session.execute(stmt_pending_ingest).scalar_one())
            current_time = datetime.now(timezone.utc)

            stmt_running = select(func.count(EvolutionJob.id)).where(
                EvolutionJob.status == JobStatus.RUNNING,
            )
            running_jobs = int(session.execute(stmt_running).scalar_one())

            stmt_stale_running = (
                select(func.count(EvolutionJob.id))
                .where(EvolutionJob.status == JobStatus.RUNNING)
                .where(EvolutionJob.lease_expires_at.is_not(None))
                .where(EvolutionJob.lease_expires_at < current_time)
            )
            stale_running_jobs = int(session.execute(stmt_stale_running).scalar_one())

            stmt_running_without_lease = (
                select(func.count(EvolutionJob.id))
                .where(EvolutionJob.status == JobStatus.RUNNING)
                .where(
                    or_(
                        EvolutionJob.run_token.is_(None),
                        EvolutionJob.worker_id.is_(None),
                        EvolutionJob.lease_expires_at.is_(None),
                    )
                )
            )
            running_without_lease_jobs = int(session.execute(stmt_running_without_lease).scalar_one())

            lease_error_norm = func.lower(func.trim(func.coalesce(EvolutionJob.last_error, "")))
            stmt_recovery_exhausted = (
                select(func.count(EvolutionJob.id))
                .where(EvolutionJob.status == JobStatus.FAILED)
                .where(
                    EvolutionJob.recovery_count
                    > int(settings.scheduler_stale_running_max_recovery_attempts),
                )
                .where(lease_error_norm.like("lease expired after missing heartbeat;%"))
            )
            recovery_exhausted_failed_jobs = int(session.execute(stmt_recovery_exhausted).scalar_one())

            best_commit: dict[str, object] | None = None
            metric_name = str(settings.mapelites_fitness_metric or "").strip()
            if metric_name:
                order_column = (
                    Metric.value.desc()
                    if bool(settings.mapelites_fitness_higher_is_better)
                    else Metric.value.asc()
                )
                conditions = [Metric.name == metric_name]
                root_commit = str(getattr(instance, "root_commit_hash", "") or "").strip()
                if root_commit:
                    conditions.append(CommitCard.commit_hash != root_commit)
                stmt_best = (
                    select(
                        CommitCard.commit_hash,
                        CommitCard.subject,
                        CommitCard.island_id,
                        Metric.value,
                        CommitCard.created_at,
                    )
                    .join(Metric, Metric.commit_card_id == CommitCard.id)
                    .where(*conditions)
                    .order_by(order_column)
                    .limit(1)
                )
                row = session.execute(stmt_best).first()
                if row:
                    commit_hash, subject, best_island, fitness_value, created_at = row
                    best_commit = {
                        "commit_hash": str(commit_hash),
                        "subject": str(subject),
                        "island_id": str(best_island) if best_island is not None else None,
                        "metric": metric_name,
                        "fitness": float(fitness_value) if fitness_value is not None else None,
                        "created_at": created_at.isoformat() if created_at is not None else None,
                    }

            instance_payload: dict[str, object] = {
                "experiment_id_raw": str(getattr(instance, "experiment_id_raw", "") or ""),
                "experiment_uuid": str(getattr(instance, "experiment_uuid", "") or ""),
                "root_commit_hash": str(getattr(instance, "root_commit_hash", "") or ""),
                "repository_slug": getattr(instance, "repository_slug", None),
                "repository_canonical_origin": getattr(instance, "repository_canonical_origin", None),
            }
            jobs_payload: dict[str, int] = {
                "unfinished": unfinished_jobs,
                "pending_ingestion": pending_ingestion_jobs,
            }
            lease_payload: dict[str, int] = {
                "lease_ttl_seconds": int(settings.worker_job_lease_ttl_seconds),
                "heartbeat_interval_seconds": int(settings.worker_job_heartbeat_interval_seconds),
                "max_recovery_attempts": int(settings.scheduler_stale_running_max_recovery_attempts),
                "running": running_jobs,
                "stale_running": stale_running_jobs,
                "running_without_lease": running_without_lease_jobs,
                "recovery_exhausted_failed": recovery_exhausted_failed_jobs,
            }
    except typer.Exit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        console.print(f"[bold red]Failed to load status[/] reason={exc}")
        raise typer.Exit(code=1) from exc

    archive_stats = _load_archive_stats_or_exit(settings=settings, island_id=effective_island)

    payload: dict[str, object] = {
        "instance": instance_payload,
        "jobs": jobs_payload,
        "job_leases": lease_payload,
        "archive": archive_stats,
        "best_commit": best_commit,
    }

    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return

    from rich.table import Table

    table = Table(title="Loreley status")
    table.add_column("field", style="bold")
    table.add_column("value")

    table.add_row("experiment_id", str(instance_payload.get("experiment_id_raw") or "n/a"))
    table.add_row("root_commit", _short_hash(str(instance_payload.get("root_commit_hash") or "")))
    repo_slug = instance_payload.get("repository_slug")
    if repo_slug:
        table.add_row("repository", str(repo_slug))
    origin = instance_payload.get("repository_canonical_origin")
    if origin:
        table.add_row("origin", str(origin))

    table.add_section()
    table.add_row("unfinished_jobs", str(unfinished_jobs))
    table.add_row("pending_ingestion", str(pending_ingestion_jobs))

    table.add_section()
    table.add_row("running_jobs", str(lease_payload["running"]))
    table.add_row("stale_running", str(lease_payload["stale_running"]))
    table.add_row("running_without_lease", str(lease_payload["running_without_lease"]))
    table.add_row("recovery_exhausted_failed", str(lease_payload["recovery_exhausted_failed"]))
    table.add_row("lease_ttl_seconds", str(lease_payload["lease_ttl_seconds"]))
    table.add_row("heartbeat_interval_seconds", str(lease_payload["heartbeat_interval_seconds"]))
    table.add_row("max_recovery_attempts", str(lease_payload["max_recovery_attempts"]))

    table.add_section()
    table.add_row("island_id", str(archive_stats.get("island_id") or effective_island))
    occupied = archive_stats.get("occupied")
    cells = archive_stats.get("cells")
    if isinstance(occupied, (int, float)) and isinstance(cells, (int, float)) and int(cells) > 0:
        table.add_row("occupied", f"{int(occupied)}/{int(cells)}")
    else:
        table.add_row("occupied", "n/a")
    coverage = archive_stats.get("coverage")
    if isinstance(coverage, (int, float)):
        table.add_row("coverage", f"{float(coverage) * 100.0:.2f}%")
    else:
        table.add_row("coverage", "n/a")
    qd_score = archive_stats.get("qd_score")
    if isinstance(qd_score, (int, float)):
        table.add_row("qd_score", f"{float(qd_score):.6f}")
    else:
        table.add_row("qd_score", "n/a")
    norm_qd_score = archive_stats.get("norm_qd_score")
    if isinstance(norm_qd_score, (int, float)):
        table.add_row("norm_qd_score", f"{float(norm_qd_score):.6f}")
    else:
        table.add_row("norm_qd_score", "n/a")

    table.add_section()
    if best_commit:
        table.add_row("best_commit", _short_hash(str(best_commit.get("commit_hash") or "")))
        table.add_row("best_metric", str(best_commit.get("metric") or "n/a"))
        fitness = best_commit.get("fitness")
        if isinstance(fitness, (int, float)):
            table.add_row("best_fitness", f"{float(fitness):.6f}")
        else:
            table.add_row("best_fitness", "n/a")
        best_island = best_commit.get("island_id")
        if best_island:
            table.add_row("best_island", str(best_island))
        subject = best_commit.get("subject")
        if subject:
            table.add_row("best_subject", str(subject))
    else:
        table.add_row("best_commit", "n/a")

    console.print(table)


@jobs_app.command("retry")
def retry_job(
    ctx: typer.Context,
    job_id: str | None = typer.Argument(None, help="Evolution job UUID to retry."),
    failed_stale: bool = typer.Option(
        False,
        "--failed-stale",
        help="Retry FAILED jobs that exhausted the stale-lease recovery budget.",
        show_default=True,
    ),
    retry_all: bool = typer.Option(
        False,
        "--all",
        help="When used with --failed-stale, retry all matching jobs.",
        show_default=True,
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="When used with --failed-stale, retry up to N matching jobs.",
        show_default=False,
    ),
    reason: str = typer.Option(
        "manual retry requested via CLI",
        "--reason",
        help="Reason written to last_error while the job is requeued.",
        show_default=True,
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print the retry result as JSON.",
        show_default=True,
    ),
) -> None:
    """Requeue a failed or stuck evolution job."""
    settings = _load_settings_or_exit()
    _configure_logging_or_exit(settings=settings, role="jobs", override_level=_get_log_level(ctx))

    raw_job_id = str(job_id or "").strip()
    if raw_job_id and failed_stale:
        console.print("[bold red]Choose either a job id or --failed-stale[/]")
        raise typer.Exit(code=1)
    if raw_job_id and (retry_all or limit is not None):
        console.print("[bold red]Do not combine a job id with --all or --limit[/]")
        raise typer.Exit(code=1)
    if not raw_job_id and not failed_stale:
        console.print("[bold red]Provide a job id or use --failed-stale[/]")
        raise typer.Exit(code=1)
    if failed_stale:
        if retry_all and limit is not None:
            console.print("[bold red]Choose either --all or --limit with --failed-stale[/]")
            raise typer.Exit(code=1)
        if not retry_all and limit is None:
            console.print("[bold red]Use --all or --limit with --failed-stale[/]")
            raise typer.Exit(code=1)

    try:
        from sqlalchemy import select

        from loreley.db.base import session_scope
        from loreley.db.models import EvolutionJob, JobStatus

        now = datetime.now(timezone.utc)
        with session_scope() as session:
            if raw_job_id:
                try:
                    job_uuid = uuid.UUID(raw_job_id)
                except ValueError as exc:
                    console.print(f"[bold red]Invalid job id[/] value={raw_job_id}")
                    raise typer.Exit(code=1) from exc
                job = session.get(EvolutionJob, job_uuid)
                if job is None:
                    console.print(f"[bold red]Job not found[/] id={job_uuid}")
                    raise typer.Exit(code=1)
                retryable, lease_state = _job_retry_state(job=job, now=now)
                if not retryable:
                    console.print(
                        "[bold red]Only failed or stuck RUNNING jobs can be retried[/] "
                        f"id={job_uuid} status={job.status} lease_state={lease_state or 'n/a'}",
                    )
                    raise typer.Exit(code=1)
                payload = _retry_job_row(job=job, reason=reason, now=now)
            else:
                from sqlalchemy import func

                stmt = (
                    select(EvolutionJob)
                    .where(
                        *_failed_stale_job_conditions(
                            EvolutionJob=EvolutionJob,
                            JobStatus=JobStatus,
                            func=func,
                            max_recovery_attempts=int(
                                settings.scheduler_stale_running_max_recovery_attempts,
                            ),
                        )
                    )
                    .order_by(
                        EvolutionJob.completed_at.desc().nullslast(),
                        EvolutionJob.created_at.desc(),
                    )
                )
                if not retry_all and limit is not None:
                    stmt = stmt.limit(int(limit))
                rows = list(session.execute(stmt).scalars())
                retried_jobs = [_retry_job_row(job=row, reason=reason, now=now) for row in rows]
                payload = {
                    "filters": {
                        "failed_stale": True,
                        "all": bool(retry_all),
                        "limit": None if retry_all else int(limit or 0),
                    },
                    "count": len(retried_jobs),
                    "retried_jobs": retried_jobs,
                }
    except typer.Exit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        target = raw_job_id or "failed_stale"
        console.print(f"[bold red]Failed to retry job[/] target={target} reason={exc}")
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if raw_job_id:
        console.print(
            "[bold green]Retried job[/] "
            f"id={payload['job_id']} status={payload['previous_status']}->{payload['new_status']} "
            f"recovery_count_reset_from={payload['recovery_count_reset_from']}",
        )
        return

    console.print(
        "[bold green]Retried jobs[/] "
        f"count={payload['count']} failed_stale=true all={payload['filters']['all']} "
        f"limit={payload['filters']['limit']}",
    )


@jobs_app.command("inspect")
def inspect_job(
    ctx: typer.Context,
    job_id: str = typer.Argument(..., help="Evolution job UUID to inspect."),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print the job payload as JSON.",
        show_default=True,
    ),
) -> None:
    """Print detailed status and lease information for one evolution job."""
    settings = _load_settings_or_exit()
    _configure_logging_or_exit(settings=settings, role="jobs", override_level=_get_log_level(ctx))

    raw_job_id = str(job_id).strip()
    try:
        job_uuid = uuid.UUID(raw_job_id)
    except ValueError as exc:
        console.print(f"[bold red]Invalid job id[/] value={raw_job_id}")
        raise typer.Exit(code=1) from exc

    try:
        from loreley.db.base import session_scope
        from loreley.db.models import EvolutionJob

        with session_scope() as session:
            job = session.get(EvolutionJob, job_uuid)
            if job is None:
                console.print(f"[bold red]Job not found[/] id={job_uuid}")
                raise typer.Exit(code=1)

            now = datetime.now(timezone.utc)
            payload = {
                **_job_summary_payload(job=job, now=now),
                "scheduled_at": _iso_or_none(getattr(job, "scheduled_at", None)),
                "started_at": _iso_or_none(getattr(job, "started_at", None)),
                "heartbeat_at": _iso_or_none(getattr(job, "heartbeat_at", None)),
                "lease_expires_at": _iso_or_none(getattr(job, "lease_expires_at", None)),
                "lease": _job_lease_payload(job=job, now=now),
            }
    except typer.Exit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        console.print(f"[bold red]Failed to inspect job[/] id={job_uuid} reason={exc}")
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return

    from rich.table import Table

    table = Table(title=f"Evolution job {payload['job_id']}")
    table.add_column("field", style="bold")
    table.add_column("value")
    for key in (
        "status",
        "base_commit_hash",
        "island_id",
        "recovery_count",
        "result_commit_hash",
        "created_at",
        "scheduled_at",
        "started_at",
        "completed_at",
        "last_error",
    ):
        table.add_row(str(key), _display_or_na(payload.get(key)))
    table.add_section()
    lease = payload["lease"]
    for key in ("state", "worker_id", "run_token", "heartbeat_at", "lease_expires_at"):
        table.add_row(f"lease.{key}", _display_or_na(lease.get(key)))
    console.print(table)


@jobs_app.command("ls")
def list_jobs(
    ctx: typer.Context,
    failed_stale: bool = typer.Option(
        False,
        "--failed-stale",
        help="Only show FAILED jobs that exhausted the stale-lease recovery budget.",
        show_default=True,
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        min=1,
        help="Maximum number of jobs to return.",
        show_default=True,
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print the job list as JSON.",
        show_default=True,
    ),
) -> None:
    """List recent evolution jobs, with optional stale-failure filtering."""
    settings = _load_settings_or_exit()
    _configure_logging_or_exit(settings=settings, role="jobs", override_level=_get_log_level(ctx))

    try:
        from sqlalchemy import func, select

        from loreley.db.base import session_scope
        from loreley.db.models import EvolutionJob, JobStatus

        with session_scope() as session:
            stmt = select(EvolutionJob)
            if failed_stale:
                stmt = stmt.where(
                    *_failed_stale_job_conditions(
                        EvolutionJob=EvolutionJob,
                        JobStatus=JobStatus,
                        func=func,
                        max_recovery_attempts=int(
                            settings.scheduler_stale_running_max_recovery_attempts,
                        ),
                    )
                )
            stmt = (
                stmt.order_by(
                    EvolutionJob.completed_at.desc().nullslast(),
                    EvolutionJob.created_at.desc(),
                )
                .limit(int(limit))
            )
            rows = list(session.execute(stmt).scalars())
            now = datetime.now(timezone.utc)
            jobs = [_job_summary_payload(job=row, now=now) for row in rows]
    except Exception as exc:  # pragma: no cover - defensive
        console.print(f"[bold red]Failed to list jobs[/] reason={exc}")
        raise typer.Exit(code=1) from exc

    payload = {
        "filters": {"failed_stale": bool(failed_stale)},
        "jobs": jobs,
    }
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return

    from rich.table import Table

    table = Table(title="Evolution jobs")
    table.add_column("job_id", style="bold")
    table.add_column("status")
    table.add_column("lease")
    table.add_column("recovery")
    table.add_column("base_commit")
    table.add_column("completed_at")
    for job in jobs:
        table.add_row(
            str(job["job_id"]),
            str(job["status"]),
            str(job["lease_state"]),
            str(job["recovery_count"]),
            str(job["base_commit_hash"] or "n/a"),
            str(job["completed_at"] or "n/a"),
        )
    console.print(table)


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
    effective_island = _resolve_effective_island(settings=settings, island_id=island_id)
    stats = _load_archive_stats_or_exit(settings=settings, island_id=effective_island)

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

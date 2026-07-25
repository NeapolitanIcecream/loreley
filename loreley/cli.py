from __future__ import annotations

"""Unified CLI for Loreley.

This CLI is designed to:
- provide a single entrypoint (`loreley ...`)
- run preflight checks before starting long-running processes
"""

from contextlib import nullcontext, redirect_stdout
from datetime import datetime
import os
import sys
import json
from enum import Enum
from typing import Any, Sequence
import uuid

import click
import typer
from rich.console import Console

from loreley.config import (
    Settings,
    get_settings,
    resolve_default_island_id,
    resolve_objective_contract,
)
from loreley.core.candidate_fate import CandidateFate, derive_candidate_fate
from loreley.core.job_retry import (
    db_utc_now,
    failed_stale_job_conditions,
    job_lease_payload,
    job_retry_state,
    load_failed_stale_retry_rows,
    retry_failed_stale_jobs_payload,
    retry_job_row,
)
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
db_app = typer.Typer(help="Inspect and migrate the Loreley database schema.")
app.add_typer(config_app, name="config")
app.add_typer(db_app, name="db")
app.add_typer(jobs_app, name="jobs")
archive_app = typer.Typer(help="Inspect MAP-Elites archives.")
app.add_typer(archive_app, name="archive")
embedding_cache_app = typer.Typer(help="Manage repo-state embedding cache manifests and imports.")
app.add_typer(embedding_cache_app, name="embedding-cache")


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
        from loreley.core.map_elites.manager import MapElitesManager

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


def _db_utc_now(session: Any) -> datetime:
    return db_utc_now(session)


def _job_lease_payload(*, job: Any, now: datetime) -> dict[str, object]:
    return job_lease_payload(job=job, now=now)


def _candidate_fate_fields(*, job: Any, candidate_fate: CandidateFate | None) -> dict[str, object]:
    fate = candidate_fate or derive_candidate_fate(job=job)
    return fate.as_dict()


def _job_summary_payload(
    *,
    job: Any,
    now: datetime,
    candidate_fate: CandidateFate | None = None,
) -> dict[str, object]:
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
        **_candidate_fate_fields(job=job, candidate_fate=candidate_fate),
    }


def _job_detail_payload(
    *,
    job: Any,
    now: datetime,
    candidate_fate: CandidateFate | None = None,
) -> dict[str, object]:
    return {
        **_job_summary_payload(job=job, now=now, candidate_fate=candidate_fate),
        "scheduled_at": _iso_or_none(getattr(job, "scheduled_at", None)),
        "started_at": _iso_or_none(getattr(job, "started_at", None)),
        "heartbeat_at": _iso_or_none(getattr(job, "heartbeat_at", None)),
        "lease_expires_at": _iso_or_none(getattr(job, "lease_expires_at", None)),
        "lease": _job_lease_payload(job=job, now=now),
    }


def _instance_status_payload(instance: Any) -> dict[str, object]:
    return {
        "experiment_id_raw": str(getattr(instance, "experiment_id_raw", "") or ""),
        "experiment_uuid": str(getattr(instance, "experiment_uuid", "") or ""),
        "root_commit_hash": str(getattr(instance, "root_commit_hash", "") or ""),
        "repository_slug": getattr(instance, "repository_slug", None),
        "repository_canonical_origin": getattr(instance, "repository_canonical_origin", None),
    }


def _jobs_status_payload(*, unfinished_jobs: int, pending_ingestion_jobs: int) -> dict[str, int]:
    return {
        "unfinished": int(unfinished_jobs),
        "pending_ingestion": int(pending_ingestion_jobs),
    }


def _lease_status_payload(
    *,
    settings: Settings,
    running_jobs: int,
    stale_running_jobs: int,
    running_without_lease_jobs: int,
    recovery_exhausted_failed_jobs: int,
) -> dict[str, int]:
    return {
        "lease_ttl_seconds": int(settings.worker_job_lease_ttl_seconds),
        "heartbeat_interval_seconds": int(settings.worker_job_heartbeat_interval_seconds),
        "max_recovery_attempts": int(settings.scheduler_stale_running_max_recovery_attempts),
        "running": int(running_jobs),
        "stale_running": int(stale_running_jobs),
        "running_without_lease": int(running_without_lease_jobs),
        "recovery_exhausted_failed": int(recovery_exhausted_failed_jobs),
    }


def _best_commit_status_payload(*, row: Any, metric_name: str) -> dict[str, object]:
    commit_hash, subject, best_island, primary_value, created_at = row
    return {
        "commit_hash": str(commit_hash),
        "subject": str(subject),
        "island_id": str(best_island) if best_island is not None else None,
        "primary_metric": metric_name,
        "primary_value": float(primary_value) if primary_value is not None else None,
        "created_at": created_at.isoformat() if created_at is not None else None,
    }


def _baseline_status_payload(*, row: Any | None) -> dict[str, object] | None:
    if row is None:
        return None
    return {
        "campaign_baseline_id": str(getattr(row, "id", "")) if getattr(row, "id", None) else None,
        "baseline_key_hash": getattr(row, "baseline_key_hash", None),
        "root_baseline_commit": getattr(row, "root_commit_hash", None),
        "root_baseline_metric": getattr(row, "primary_metric_name", None),
        "root_baseline_value": getattr(row, "metric_value", None),
        "root_baseline_direction": (
            "higher_is_better"
            if bool(getattr(row, "primary_metric_higher_is_better", True))
            else "lower_is_better"
        ),
        "root_baseline_status": getattr(row, "status", None),
        "baseline_campaign_program_hash": getattr(row, "campaign_program_hash", None),
        "failure_kind": getattr(row, "failure_kind", None),
        "failure_summary": getattr(row, "failure_summary", None),
    }


def _load_status_baseline_payload(*, session: Any, settings: Settings) -> dict[str, object] | None:
    from loreley.scheduler.baselines import (
        load_latest_matching_baseline,
        resolve_status_campaign_program_hash,
    )

    campaign_resolution = resolve_status_campaign_program_hash(
        session=session,
        settings=settings,
    )
    if not campaign_resolution.known:
        return None
    baseline_row = load_latest_matching_baseline(
        session=session,
        settings=settings,
        campaign_program_hash=campaign_resolution.campaign_program_hash,
    )
    return _baseline_status_payload(row=baseline_row)


def _status_response_payload(
    *,
    instance_payload: dict[str, object],
    jobs_payload: dict[str, int],
    lease_payload: dict[str, int],
    archive_stats: dict[str, object],
    best_commit: dict[str, object] | None,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "instance": instance_payload,
        "jobs": jobs_payload,
        "job_leases": lease_payload,
        "archive": archive_stats,
        "best_commit": best_commit,
        "baseline": baseline,
    }


def _failed_stale_job_conditions(
    *,
    EvolutionJob: Any,
    JobStatus: Any,
    func: Any,
    max_recovery_attempts: int,
) -> tuple[Any, ...]:
    return failed_stale_job_conditions(
        EvolutionJob=EvolutionJob,
        JobStatus=JobStatus,
        func=func,
        max_recovery_attempts=max_recovery_attempts,
    )


def _count_status_rows(session: Any, stmt: Any) -> int:
    return int(session.execute(stmt).scalar_one())


def _status_query_deps() -> tuple[Any, Any, Any, Any, Any]:
    from sqlalchemy import func, or_, select

    from loreley.db.models import EvolutionJob, JobStatus

    return select, func, or_, EvolutionJob, JobStatus


def _count_unfinished_jobs(
    *,
    session: Any,
    EvolutionJob: Any,
    JobStatus: Any,
    select: Any,
    func: Any,
) -> int:
    return _count_status_rows(
        session,
        select(func.count(EvolutionJob.id)).where(
            EvolutionJob.status.in_((JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING)),
        ),
    )


def _count_pending_ingestion_jobs(
    *,
    session: Any,
    EvolutionJob: Any,
    JobStatus: Any,
    select: Any,
    func: Any,
) -> int:
    status_norm = func.lower(func.trim(func.coalesce(EvolutionJob.ingestion_status, "")))
    commit_norm = func.trim(func.coalesce(EvolutionJob.result_commit_hash, ""))
    stmt = (
        select(func.count(EvolutionJob.id))
        .where(EvolutionJob.status == JobStatus.SUCCEEDED)
        .where(status_norm.not_in(("succeeded", "skipped")))
        .where(commit_norm != "")
    )
    return _count_status_rows(session, stmt)


def _count_running_jobs(
    *,
    session: Any,
    EvolutionJob: Any,
    JobStatus: Any,
    select: Any,
    func: Any,
) -> int:
    stmt = select(func.count(EvolutionJob.id)).where(EvolutionJob.status == JobStatus.RUNNING)
    return _count_status_rows(session, stmt)


def _count_stale_running_jobs(
    *,
    session: Any,
    EvolutionJob: Any,
    JobStatus: Any,
    select: Any,
    func: Any,
    current_time: datetime,
) -> int:
    stmt = (
        select(func.count(EvolutionJob.id))
        .where(EvolutionJob.status == JobStatus.RUNNING)
        .where(EvolutionJob.lease_expires_at.is_not(None))
        .where(EvolutionJob.lease_expires_at < current_time)
    )
    return _count_status_rows(session, stmt)


def _count_running_without_lease_jobs(
    *,
    session: Any,
    EvolutionJob: Any,
    JobStatus: Any,
    select: Any,
    func: Any,
    or_: Any,
) -> int:
    stmt = (
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
    return _count_status_rows(session, stmt)


def _count_recovery_exhausted_failed_jobs(
    *,
    session: Any,
    EvolutionJob: Any,
    JobStatus: Any,
    select: Any,
    func: Any,
    max_recovery_attempts: int,
) -> int:
    stmt = select(func.count(EvolutionJob.id)).where(
        *_failed_stale_job_conditions(
            EvolutionJob=EvolutionJob,
            JobStatus=JobStatus,
            func=func,
            max_recovery_attempts=max_recovery_attempts,
        )
    )
    return _count_status_rows(session, stmt)


def _load_status_job_payloads(
    *,
    session: Any,
    settings: Settings,
    current_time: datetime,
) -> tuple[dict[str, int], dict[str, int]]:
    select, func, or_, EvolutionJob, JobStatus = _status_query_deps()

    unfinished_jobs = _count_unfinished_jobs(
        session=session,
        EvolutionJob=EvolutionJob,
        JobStatus=JobStatus,
        select=select,
        func=func,
    )
    pending_ingestion_jobs = _count_pending_ingestion_jobs(
        session=session,
        EvolutionJob=EvolutionJob,
        JobStatus=JobStatus,
        select=select,
        func=func,
    )
    running_jobs = _count_running_jobs(
        session=session,
        EvolutionJob=EvolutionJob,
        JobStatus=JobStatus,
        select=select,
        func=func,
    )
    stale_running_jobs = _count_stale_running_jobs(
        session=session,
        EvolutionJob=EvolutionJob,
        JobStatus=JobStatus,
        select=select,
        func=func,
        current_time=current_time,
    )
    running_without_lease_jobs = _count_running_without_lease_jobs(
        session=session,
        EvolutionJob=EvolutionJob,
        JobStatus=JobStatus,
        select=select,
        func=func,
        or_=or_,
    )
    recovery_exhausted_failed_jobs = _count_recovery_exhausted_failed_jobs(
        session=session,
        EvolutionJob=EvolutionJob,
        JobStatus=JobStatus,
        select=select,
        func=func,
        max_recovery_attempts=int(settings.scheduler_stale_running_max_recovery_attempts),
    )

    return (
        _jobs_status_payload(
            unfinished_jobs=unfinished_jobs,
            pending_ingestion_jobs=pending_ingestion_jobs,
        ),
        _lease_status_payload(
            settings=settings,
            running_jobs=running_jobs,
            stale_running_jobs=stale_running_jobs,
            running_without_lease_jobs=running_without_lease_jobs,
            recovery_exhausted_failed_jobs=recovery_exhausted_failed_jobs,
        ),
    )


def _load_best_commit_status_payload(
    *,
    session: Any,
    settings: Settings,
    instance: Any,
    CommitCard: Any,
    MapElitesArchiveCell: Any,
    Metric: Any,
) -> dict[str, object] | None:
    from sqlalchemy import select

    primary = resolve_objective_contract(settings).primary
    metric_name = primary.name

    order_column = (
        Metric.value.desc()
        if primary.higher_is_better
        else Metric.value.asc()
    )
    conditions = [
        Metric.name == metric_name,
        MapElitesArchiveCell.island_id.in_(tuple(settings.mapelites_islands)),
    ]
    root_commit = str(getattr(instance, "root_commit_hash", "") or "").strip()
    if root_commit:
        conditions.append(CommitCard.commit_hash != root_commit)
    stmt_best = (
        select(
            CommitCard.commit_hash,
            CommitCard.subject,
            MapElitesArchiveCell.island_id,
            Metric.value,
            CommitCard.created_at,
        )
        .join(Metric, Metric.commit_card_id == CommitCard.id)
        .join(
            MapElitesArchiveCell,
            MapElitesArchiveCell.commit_hash == CommitCard.commit_hash,
        )
        .where(*conditions)
        .order_by(order_column, CommitCard.commit_hash.asc())
        .limit(1)
    )
    row = session.execute(stmt_best).first()
    if not row:
        return None
    return _best_commit_status_payload(row=row, metric_name=metric_name)


def _job_retry_state(*, job: Any, now: datetime) -> tuple[bool, str | None]:
    return job_retry_state(job=job, now=now)


def _retry_job_row(*, job: Any, reason: str, now: datetime) -> dict[str, object]:
    return retry_job_row(
        job=job,
        reason=str(reason or "").strip() or "manual retry requested via CLI",
        now=now,
    )


def _retry_failed_stale_jobs_payload(
    *,
    session: Any,
    settings: Settings,
    retry_all: bool,
    limit: int | None,
    reason: str,
    now: datetime,
) -> dict[str, object]:
    return retry_failed_stale_jobs_payload(
        session=session,
        max_recovery_attempts=int(settings.scheduler_stale_running_max_recovery_attempts),
        retry_all=retry_all,
        limit=limit,
        reason=str(reason or "").strip() or "manual retry requested via CLI",
        now=now,
    )


def _load_failed_stale_retry_rows(
    *,
    session: Any,
    max_recovery_attempts: int,
    retry_all: bool,
    limit: int | None,
) -> list[Any]:
    return load_failed_stale_retry_rows(
        session=session,
        max_recovery_attempts=max_recovery_attempts,
        retry_all=retry_all,
        limit=limit,
    )


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
    processes: int = typer.Option(
        1,
        "--processes",
        "-p",
        min=1,
        help="Number of isolated worker processes (one thread each).",
        show_default=True,
    ),
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
        processes=int(processes),
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
    """Run the UI API (FastAPI via uvicorn)."""
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


def _db_status_payload(status: Any) -> dict[str, object]:
    return {
        "schema_version": status.schema_version,
        "target": status.target_version,
        "state": status.state,
        "needs_migration": status.needs_migration,
        "detail": status.detail,
    }


def _print_db_error_and_exit(exc: Exception) -> None:
    console.print(f"[bold red]Database schema error[/] {exc}")
    raise typer.Exit(code=1) from exc


def _embedding_cache_error_and_exit(exc: Exception) -> None:
    console.print(f"[bold red]Embedding cache error[/] {exc}")
    raise typer.Exit(code=1) from exc


def _redirect_stdout_for_json(json_output: bool):
    return redirect_stdout(sys.stderr) if json_output else nullcontext()


@db_app.command("current")
def db_current(
    ctx: typer.Context,
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print schema status as JSON.",
        show_default=True,
    ),
) -> None:
    """Print the current database schema marker."""

    with redirect_stdout(sys.stderr):
        settings = _load_settings_or_exit()
        _configure_logging_or_exit(settings=settings, role="db", override_level=_get_log_level(ctx))
        try:
            from loreley.db.base import INSTANCE_SCHEMA_VERSION, get_engine
            from loreley.db.migrations.runner import describe_schema

            status = describe_schema(engine=get_engine(), target_version=INSTANCE_SCHEMA_VERSION)
        except Exception as exc:  # pragma: no cover - defensive
            _print_db_error_and_exit(exc)

    payload = _db_status_payload(status)
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    typer.echo(
        "schema_version={} target={} state={} needs_migration={}".format(
            "none" if status.schema_version is None else status.schema_version,
            status.target_version,
            status.state,
            str(bool(status.needs_migration)).lower(),
        )
    )


@db_app.command("migrate")
def db_migrate(ctx: typer.Context) -> None:
    """Migrate the database schema to the current Loreley version."""

    settings = _load_settings_or_exit()
    _configure_logging_or_exit(settings=settings, role="db", override_level=_get_log_level(ctx))
    try:
        from loreley.db.base import INSTANCE_SCHEMA_VERSION, get_engine
        from loreley.db.migrations.runner import ensure_schema_current, validate_database_schema

        engine = get_engine()
        result = ensure_schema_current(
            engine=engine,
            settings=settings,
            target_version=INSTANCE_SCHEMA_VERSION,
            auto_migrate=True,
        )
        validate_database_schema(
            engine=engine,
            settings=settings,
            target_version=INSTANCE_SCHEMA_VERSION,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _print_db_error_and_exit(exc)

    applied = ",".join(str(version) for version in result.applied_versions) or "none"
    from_version = "none" if result.from_version is None else str(result.from_version)
    typer.echo(
        "from={} to={} applied={} fresh={}".format(
            from_version,
            result.to_version,
            applied,
            str(bool(result.fresh_database)).lower(),
        )
    )


@db_app.command("validate")
def db_validate(ctx: typer.Context) -> None:
    """Validate that the database schema is current and usable."""

    settings = _load_settings_or_exit()
    _configure_logging_or_exit(settings=settings, role="db", override_level=_get_log_level(ctx))
    try:
        from loreley.db.base import INSTANCE_SCHEMA_VERSION, get_engine
        from loreley.db.migrations.runner import validate_database_schema

        status = validate_database_schema(
            engine=get_engine(),
            settings=settings,
            target_version=INSTANCE_SCHEMA_VERSION,
        )
    except Exception as exc:
        _print_db_error_and_exit(exc)
    typer.echo(f"valid schema_version={status.schema_version} target={status.target_version}")


@embedding_cache_app.command("attest")
def embedding_cache_attest(
    ctx: typer.Context,
    database_url: str | None = typer.Option(
        None,
        "--database-url",
        help="Database URL to attest; defaults to the current DATABASE_URL.",
        show_default=False,
    ),
    from_current_settings: bool = typer.Option(
        False,
        "--from-current-settings",
        help="Use current MAPELITES_* and OpenAI-compatible settings as the cache semantics.",
        show_default=True,
    ),
    fingerprint: str | None = typer.Option(
        None,
        "--fingerprint",
        help="Expected current-settings fingerprint; attestation fails if it differs.",
        show_default=False,
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print the attestation result as JSON.",
        show_default=True,
    ),
) -> None:
    """Attach a manifest to a legacy repo-state file embedding cache."""

    with _redirect_stdout_for_json(json_output):
        settings = _load_settings_or_exit()
        _configure_logging_or_exit(
            settings=settings,
            role="embedding-cache",
            override_level=_get_log_level(ctx),
        )
        if not from_current_settings and not str(fingerprint or "").strip():
            console.print("[bold red]Provide --from-current-settings or --fingerprint[/]")
            raise typer.Exit(code=1)

        try:
            from loreley.core.map_elites.embedding_cache_manifest import (
                attest_repo_state_file_embedding_cache,
            )

            result = attest_repo_state_file_embedding_cache(
                settings=settings,
                dsn=str(database_url or settings.database_dsn),
                expected_fingerprint=fingerprint,
            )
        except Exception as exc:
            _embedding_cache_error_and_exit(exc)

    payload = result.as_dict()
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    typer.echo(
        "attested fingerprint={fingerprint} source={manifest_source} rows={cache_rows} "
        "model={embedding_model} dimensions={dimensions} dsn={dsn}".format(**payload)
    )


@embedding_cache_app.command("import")
def embedding_cache_import(
    ctx: typer.Context,
    source_dsn: str = typer.Option(
        ...,
        "--source-dsn",
        help="Source database URL containing a compatible attested/generated embedding cache manifest.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print the import result as JSON.",
        show_default=True,
    ),
) -> None:
    """Import compatible repo-state file embedding cache rows into the current DB."""

    with _redirect_stdout_for_json(json_output):
        settings = _load_settings_or_exit()
        _configure_logging_or_exit(
            settings=settings,
            role="embedding-cache",
            override_level=_get_log_level(ctx),
        )
        try:
            from loreley.core.map_elites.embedding_cache_manifest import (
                import_repo_state_file_embedding_cache_from_dsn,
            )
            from loreley.db.base import ensure_database_schema

            ensure_database_schema(settings=settings, validate_marker=False)
            result = import_repo_state_file_embedding_cache_from_dsn(
                settings=settings,
                source_dsn=str(source_dsn),
            )
        except Exception as exc:
            _embedding_cache_error_and_exit(exc)

    payload = result.as_dict()
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    typer.echo(
        "imported source_rows={source_rows} inserted_rows={inserted_rows} "
        "already_present_rows={already_present_rows} skipped_rows={skipped_rows} "
        "fingerprint={fingerprint} source_manifest={source_manifest} "
        "target_manifest={target_manifest}".format(**payload)
    )


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
        from loreley.db.base import session_scope
        from loreley.db.models import (
            CommitCard,
            InstanceMetadata,
            MapElitesArchiveCell,
            Metric,
        )

        with session_scope() as session:
            instance = session.get(InstanceMetadata, 1)
            if instance is None:
                from loreley.db.instance import INIT_DB_HINT

                console.print(f"[bold red]Instance metadata is missing[/] {INIT_DB_HINT}")
                raise typer.Exit(code=1)

            current_time = _db_utc_now(session)
            jobs_payload, lease_payload = _load_status_job_payloads(
                session=session,
                settings=settings,
                current_time=current_time,
            )
            best_commit = _load_best_commit_status_payload(
                session=session,
                settings=settings,
                instance=instance,
                CommitCard=CommitCard,
                MapElitesArchiveCell=MapElitesArchiveCell,
                Metric=Metric,
            )
            baseline = _load_status_baseline_payload(
                session=session,
                settings=settings,
            )
            instance_payload = _instance_status_payload(instance)
    except typer.Exit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        console.print(f"[bold red]Failed to load status[/] reason={exc}")
        raise typer.Exit(code=1) from exc

    archive_stats = _load_archive_stats_or_exit(settings=settings, island_id=effective_island)

    payload = _status_response_payload(
        instance_payload=instance_payload,
        jobs_payload=jobs_payload,
        lease_payload=lease_payload,
        archive_stats=archive_stats,
        best_commit=best_commit,
        baseline=baseline,
    )

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
    table.add_row("unfinished_jobs", str(jobs_payload["unfinished"]))
    table.add_row("pending_ingestion", str(jobs_payload["pending_ingestion"]))

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
    elites = archive_stats.get("elites")
    table.add_row("elites", str(int(elites)) if isinstance(elites, (int, float)) else "n/a")
    table.add_row(
        "objectives",
        str(archive_stats.get("objective_count") or "n/a"),
    )

    table.add_section()
    if best_commit:
        table.add_row("best_commit", _short_hash(str(best_commit.get("commit_hash") or "")))
        table.add_row(
            "primary_metric",
            str(best_commit.get("primary_metric") or "n/a"),
        )
        primary_value = best_commit.get("primary_value")
        if isinstance(primary_value, (int, float)):
            table.add_row("best_primary_value", f"{float(primary_value):.6f}")
        else:
            table.add_row("best_primary_value", "n/a")
        best_island = best_commit.get("island_id")
        if best_island:
            table.add_row("best_island", str(best_island))
        subject = best_commit.get("subject")
        if subject:
            table.add_row("best_subject", str(subject))
    else:
        table.add_row("best_commit", "n/a")

    if baseline:
        table.add_section()
        table.add_row("root_baseline_status", str(baseline.get("root_baseline_status") or "n/a"))
        table.add_row("root_baseline_metric", str(baseline.get("root_baseline_metric") or "n/a"))
        baseline_value = baseline.get("root_baseline_value")
        if isinstance(baseline_value, (int, float)):
            table.add_row("root_baseline_value", f"{float(baseline_value):.6f}")
        else:
            table.add_row("root_baseline_value", "n/a")
        table.add_row("baseline_key", _short_hash(str(baseline.get("baseline_key_hash") or "")))
        if baseline.get("failure_kind"):
            table.add_row("baseline_failure", str(baseline.get("failure_kind")))

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
        from loreley.db.base import session_scope
        from loreley.db.models import EvolutionJob

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
                now = _db_utc_now(session)
                retryable, lease_state = _job_retry_state(job=job, now=now)
                if not retryable:
                    console.print(
                        "[bold red]Only failed or stuck RUNNING jobs can be retried[/] "
                        f"id={job_uuid} status={job.status} lease_state={lease_state or 'n/a'}",
                    )
                    raise typer.Exit(code=1)
                payload = _retry_job_row(job=job, reason=reason, now=now)
            else:
                now = _db_utc_now(session)
                payload = _retry_failed_stale_jobs_payload(
                    session=session,
                    settings=settings,
                    retry_all=bool(retry_all),
                    limit=limit,
                    reason=reason,
                    now=now,
                )
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
        from loreley.api.services.candidate_fates import load_candidate_fates_for_jobs
        from loreley.db.base import session_scope
        from loreley.db.models import EvolutionJob

        with session_scope() as session:
            job = session.get(EvolutionJob, job_uuid)
            if job is None:
                console.print(f"[bold red]Job not found[/] id={job_uuid}")
                raise typer.Exit(code=1)

            now = _db_utc_now(session)
            fates = load_candidate_fates_for_jobs([job])
            payload = _job_detail_payload(
                job=job,
                now=now,
                candidate_fate=fates.get(str(job.id)),
            )
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
        "candidate_fate_label",
        "candidate_fate_reason",
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

        from loreley.api.services.candidate_fates import load_candidate_fates_for_jobs
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
            now = _db_utc_now(session)
            fates = load_candidate_fates_for_jobs(rows)
            jobs = [
                _job_summary_payload(
                    job=row,
                    now=now,
                    candidate_fate=fates.get(str(getattr(row, "id", "") or "")),
                )
                for row in rows
            ]
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
    table.add_column("fate")
    table.add_column("base_commit")
    table.add_column("completed_at")
    for job in jobs:
        table.add_row(
            str(job["job_id"]),
            str(job["status"]),
            str(job["lease_state"]),
            str(job["recovery_count"]),
            str(job["candidate_fate_label"] or "n/a"),
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
        "elites",
        "cells",
        "coverage",
        "objective_count",
        "front_max_size",
        "best_primary_value",
        "primary_metric_name",
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

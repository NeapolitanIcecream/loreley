"""Operator console status and background-task services."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from loguru import logger
from rich.console import Console
from sqlalchemy import func, select

from loreley.api.services.repair import repair_pool_summary
from loreley.config import Settings, get_settings
from loreley.core.campaign_program import (
    CampaignProgramLoadResult,
    CampaignProgramSnapshot,
    load_campaign_program_from_repo,
    persist_campaign_program,
)
from loreley.core.contracts import clamp_text, normalize_single_line
from loreley.core.job_retry import db_utc_now, failed_stale_job_conditions
from loreley.db.base import session_scope
from loreley.db.models import (
    EvolutionJob,
    JobStatus,
    OperatorTask,
    OperatorTaskKind,
    OperatorTaskStatus,
)
from loreley.scheduler.baselines import (
    BaselineBootstrapResult,
    BaselineBootstrapService,
    load_latest_matching_baseline,
    resolve_status_campaign_program_hash,
)

log = logger.bind(module="api.operator")


class OperatorTaskNotFoundError(RuntimeError):
    """Raised when an operator task cannot be found."""


class OperatorTaskAlreadyActiveError(RuntimeError):
    """Raised when an equivalent operator task is already pending or running."""


def operator_status(*, settings: Settings | None = None) -> dict[str, object]:
    """Return the consolidated operator console status payload."""

    active_settings = settings or get_settings()
    current_file = _current_campaign_program_file(active_settings)
    with session_scope() as session:
        generated_at = db_utc_now(session)
        campaign_resolution = resolve_status_campaign_program_hash(
            session=session,
            settings=active_settings,
        )
        baseline = _baseline_status_payload(
            load_latest_matching_baseline(
                session=session,
                settings=active_settings,
                campaign_program_hash=(
                    campaign_resolution.campaign_program_hash
                    if campaign_resolution.known
                    else None
                ),
            )
            if campaign_resolution.known
            else None
        )
        job_health = _job_health_from_session(
            session=session,
            settings=active_settings,
            current_time=generated_at,
        )
    active_hash = (
        campaign_resolution.campaign_program_hash
        if campaign_resolution.known
        else None
    )
    active_source = campaign_resolution.source_path if campaign_resolution.known else None
    persisted_hash = active_hash if str(active_source or "").startswith("database:") else None
    current_hash = current_file.get("hash")
    return {
        "campaign_program": {
            "current_file": current_file,
            "scheduler": {
                "active_hash": active_hash,
                "active_source": active_source,
                "persisted_hash": persisted_hash,
                "persisted_source": active_source if persisted_hash is not None else None,
                "current_hash": current_hash,
                "current_matches_active": _hashes_match(current_hash, active_hash),
                "change_policy": getattr(active_settings, "campaign_program_change_policy", None),
            },
        },
        "baseline": baseline,
        "repair_pool": repair_pool_summary(),
        "job_health": job_health,
        "generated_at": generated_at,
    }


def create_baseline_ensure_task(*, settings: Settings | None = None) -> OperatorTask:
    """Create a pending baseline ensure background task row."""

    active_settings = settings or get_settings()
    request_payload = {
        "kind": OperatorTaskKind.BASELINE_ENSURE.value,
        "root_commit_hash": str(getattr(active_settings, "mapelites_experiment_root_commit", "") or ""),
        "repo_root": str(_operator_repo_root(active_settings)),
        "campaign_program_change_policy": str(getattr(active_settings, "campaign_program_change_policy", "") or ""),
    }
    with session_scope() as session:
        active = _active_baseline_ensure_task(session)
        if active is not None:
            raise OperatorTaskAlreadyActiveError(
                f"Baseline ensure task already active: {active.id}."
            )
        row = OperatorTask(
            kind=OperatorTaskKind.BASELINE_ENSURE.value,
            status=OperatorTaskStatus.PENDING.value,
            request_payload=request_payload,
            result_payload={},
        )
        session.add(row)
        session.flush()
        task_id = row.id
        log.bind(task_id=str(task_id)).info("Operator baseline ensure task created")
        return row


def _active_baseline_ensure_task(session: object) -> OperatorTask | None:
    return (
        session.execute(
            select(OperatorTask)
            .where(OperatorTask.kind == OperatorTaskKind.BASELINE_ENSURE.value)
            .where(
                OperatorTask.status.in_(
                    (
                        OperatorTaskStatus.PENDING.value,
                        OperatorTaskStatus.RUNNING.value,
                    )
                )
            )
            .order_by(OperatorTask.created_at.desc(), OperatorTask.id.desc())
            .limit(1)
        )
        .scalars()
        .first()
    )


def run_baseline_ensure_task(task_id: UUID) -> None:
    """Run a baseline ensure task in the FastAPI background task pool."""

    _mark_task_running(task_id)
    try:
        settings = get_settings()
        repo_root = _operator_repo_root(settings)
        loaded = _load_campaign_program_for_task(repo_root=repo_root)
        _persist_loaded_campaign_program(loaded)
        root_commit_hash = normalize_single_line(
            str(getattr(settings, "mapelites_experiment_root_commit", "") or "")
        )
        if not root_commit_hash:
            raise RuntimeError("MAPELITES_EXPERIMENT_ROOT_COMMIT is required for baseline ensure.")
        service = BaselineBootstrapService(
            settings=settings,
            repo_root=repo_root,
            console=Console(),
        )
        result = service.ensure_or_load_baseline(
            root_commit_hash=root_commit_hash,
            campaign_program=loaded.snapshot,
        )
        _mark_task_succeeded(
            task_id,
            result_payload={
                "baseline": _baseline_result_payload(result),
                "campaign_program": _campaign_program_task_payload(loaded.snapshot),
                "repo_root": str(repo_root),
            },
        )
    except Exception as exc:  # pragma: no cover - exercised via service tests with monkeypatches
        summary = clamp_text(normalize_single_line(str(exc)), 2048)
        log.bind(task_id=str(task_id)).exception("Operator baseline ensure task failed: {}", summary)
        _mark_task_failed(task_id, error_summary=summary)


def list_operator_tasks(*, limit: int = 50) -> list[OperatorTask]:
    """Return recent operator tasks newest-first."""

    limit = max(1, min(int(limit), 200))
    with session_scope() as session:
        return list(
            session.execute(
                select(OperatorTask)
                .order_by(OperatorTask.created_at.desc(), OperatorTask.id.desc())
                .limit(limit)
            ).scalars()
        )


def get_operator_task(*, task_id: UUID) -> OperatorTask:
    """Return a single operator task."""

    with session_scope() as session:
        row = session.get(OperatorTask, task_id)
        if row is None:
            raise OperatorTaskNotFoundError("Operator task not found.")
        return row


def _mark_task_running(task_id: UUID) -> None:
    with session_scope() as session:
        row = session.get(OperatorTask, task_id)
        if row is None:
            raise OperatorTaskNotFoundError("Operator task not found.")
        row.status = OperatorTaskStatus.RUNNING.value
        row.started_at = db_utc_now(session)
        row.error_summary = None
        log.bind(task_id=str(task_id)).info("Operator baseline ensure task started")


def _mark_task_succeeded(task_id: UUID, *, result_payload: dict[str, object]) -> None:
    with session_scope() as session:
        row = session.get(OperatorTask, task_id)
        if row is None:
            raise OperatorTaskNotFoundError("Operator task not found.")
        now = db_utc_now(session)
        row.status = OperatorTaskStatus.SUCCEEDED.value
        row.result_payload = result_payload
        row.error_summary = None
        row.completed_at = now
        log.bind(task_id=str(task_id)).info("Operator baseline ensure task succeeded")


def _mark_task_failed(task_id: UUID, *, error_summary: str) -> None:
    with session_scope() as session:
        row = session.get(OperatorTask, task_id)
        if row is None:
            return
        row.status = OperatorTaskStatus.FAILED.value
        row.error_summary = error_summary
        row.completed_at = db_utc_now(session)


def _current_campaign_program_file(settings: Settings) -> dict[str, object]:
    repo_root = _operator_repo_root(settings)
    try:
        loaded = load_campaign_program_from_repo(repo_root)
    except Exception as exc:
        return {
            "found": False,
            "source_path": str(repo_root / "loreley.program.md"),
            "hash": None,
            "normalized_hash": None,
            "title": None,
            "recognized_sections": [],
            "parse_warnings": [],
            "sections": {},
            "error_summary": clamp_text(normalize_single_line(str(exc)), 512),
        }
    snapshot = loaded.snapshot
    if snapshot is None:
        return {
            "found": False,
            "source_path": str(loaded.source_path) if loaded.source_path is not None else None,
            "hash": None,
            "normalized_hash": None,
            "title": None,
            "recognized_sections": [],
            "parse_warnings": [],
            "sections": {},
            "error_summary": None,
        }
    return {
        "found": True,
        "source_path": str(loaded.source_path) if loaded.source_path is not None else snapshot.source_path,
        "hash": snapshot.raw_sha256,
        "normalized_hash": snapshot.normalized_sha256,
        "title": snapshot.title,
        "recognized_sections": list(snapshot.recognized_sections),
        "parse_warnings": [dict(item) for item in snapshot.parse_warnings],
        "sections": _campaign_program_sections(snapshot),
        "error_summary": None,
    }


def _campaign_program_sections(snapshot: CampaignProgramSnapshot) -> dict[str, object]:
    primary_metric = snapshot.primary_metric.as_dict() if snapshot.primary_metric else None
    return {
        "goal": snapshot.goal,
        "primary_metric": primary_metric,
        "correctness_gates": list(snapshot.correctness_gates),
        "editable_scope": list(snapshot.editable_scope),
        "protected_scope": list(snapshot.protected_scope),
        "evaluation_budget": list(snapshot.evaluation_budget),
        "complexity_policy": list(snapshot.complexity_policy),
        "failure_policy": list(snapshot.failure_policy),
        "logging_policy": list(snapshot.logging_policy),
        "unknown_sections": [section.as_metadata_dict() for section in snapshot.unknown_sections],
    }


def _operator_repo_root(settings: Settings) -> Path:
    raw = (
        str(getattr(settings, "scheduler_repo_root", "") or "").strip()
        or str(getattr(settings, "worker_repo_worktree", "") or "").strip()
    )
    return Path(raw).expanduser().resolve() if raw else Path.cwd().resolve()


def _load_campaign_program_for_task(
    *,
    repo_root: Path,
) -> CampaignProgramLoadResult:
    return load_campaign_program_from_repo(repo_root)


def _persist_loaded_campaign_program(loaded: CampaignProgramLoadResult) -> None:
    if loaded.snapshot is None or loaded.raw_markdown is None:
        return
    with session_scope() as session:
        persist_campaign_program(
            session=session,
            snapshot=loaded.snapshot,
            raw_markdown=loaded.raw_markdown,
        )


def _campaign_program_task_payload(snapshot: CampaignProgramSnapshot | None) -> dict[str, object]:
    if snapshot is None:
        return {
            "found": False,
            "hash": None,
        }
    return {
        "found": True,
        "hash": snapshot.raw_sha256,
        "title": snapshot.title,
        "recognized_sections": list(snapshot.recognized_sections),
        "parse_warnings": [dict(item) for item in snapshot.parse_warnings],
    }


def _baseline_result_payload(result: BaselineBootstrapResult) -> dict[str, object]:
    return {
        "can_dispatch_or_schedule": result.can_dispatch_or_schedule,
        "status": result.status,
        "policy": result.policy,
        "baseline_key_hash": result.baseline_key_hash,
        "baseline_id": result.baseline_id,
        "failure_kind": result.failure_kind,
        "failure_summary": result.failure_summary,
    }


def _baseline_status_payload(row: Any | None) -> dict[str, object] | None:
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


def _job_health_from_session(
    *,
    session: object,
    settings: Settings,
    current_time: datetime,
) -> dict[str, object]:
    unfinished = _count(
        session,
        select(func.count(EvolutionJob.id)).where(
            EvolutionJob.status.in_((JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING))
        ),
    )
    status_norm = func.lower(func.trim(func.coalesce(EvolutionJob.ingestion_status, "")))
    commit_norm = func.trim(func.coalesce(EvolutionJob.result_commit_hash, ""))
    pending_ingestion = _count(
        session,
        select(func.count(EvolutionJob.id))
        .where(EvolutionJob.status == JobStatus.SUCCEEDED)
        .where(status_norm.not_in(("succeeded", "skipped")))
        .where(commit_norm != ""),
    )
    running = _count(
        session,
        select(func.count(EvolutionJob.id)).where(EvolutionJob.status == JobStatus.RUNNING),
    )
    stale_running = _count(
        session,
        select(func.count(EvolutionJob.id))
        .where(EvolutionJob.status == JobStatus.RUNNING)
        .where(EvolutionJob.lease_expires_at.is_not(None))
        .where(EvolutionJob.lease_expires_at < current_time),
    )
    running_without_lease = _count(
        session,
        select(func.count(EvolutionJob.id)).where(
            EvolutionJob.status == JobStatus.RUNNING,
            (
                (EvolutionJob.run_token.is_(None))
                | (EvolutionJob.worker_id.is_(None))
                | (EvolutionJob.lease_expires_at.is_(None))
            ),
        ),
    )
    recovery_exhausted_failed = _count(
        session,
        select(func.count(EvolutionJob.id)).where(
            *failed_stale_job_conditions(
                EvolutionJob=EvolutionJob,
                JobStatus=JobStatus,
                func=func,
                max_recovery_attempts=int(settings.scheduler_stale_running_max_recovery_attempts),
            )
        ),
    )
    return {
        "jobs": {
            "unfinished": unfinished,
            "pending_ingestion": pending_ingestion,
        },
        "job_leases": {
            "lease_ttl_seconds": int(settings.worker_job_lease_ttl_seconds),
            "heartbeat_interval_seconds": int(settings.worker_job_heartbeat_interval_seconds),
            "max_recovery_attempts": int(settings.scheduler_stale_running_max_recovery_attempts),
            "running": running,
            "stale_running": stale_running,
            "running_without_lease": running_without_lease,
            "recovery_exhausted_failed": recovery_exhausted_failed,
        },
        "by_status": _job_group_counts(session=session, column=EvolutionJob.status),
        "by_job_kind": _job_group_counts(session=session, column=EvolutionJob.job_kind),
    }


def _count(session: object, stmt: object) -> int:
    return int(session.execute(stmt).scalar_one() or 0)


def _job_group_counts(*, session: object, column: object) -> dict[str, int]:
    rows = session.execute(select(column, func.count(EvolutionJob.id)).group_by(column)).all()
    result: dict[str, int] = {}
    for value, count in rows:
        status_value = getattr(value, "value", value)
        key = normalize_single_line(str(status_value or "unknown")) or "unknown"
        result[key] = int(count or 0)
    return result


def _hashes_match(current_hash: object, active_hash: object) -> bool | None:
    current = normalize_single_line(str(current_hash or ""))
    active = normalize_single_line(str(active_hash or ""))
    if not current and not active:
        return True
    if not current or not active:
        return False
    return current == active

"""Repair pool queries and write operations for the UI API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from loguru import logger
from sqlalchemy import and_, func, or_, select

from loreley.api.pagination import PaginationCursorError, decode_cursor, encode_cursor, normalize_pagination
from loreley.config import Settings, get_settings
from loreley.core.contracts import clamp_text, normalize_single_line
from loreley.core.repair_coordination import (
    with_repair_scheduling_lock as _with_manual_repair_schedule_lock,
)
from loreley.db.base import session_scope
from loreley.db.models import (
    CandidateCommit,
    DiagnosticCapsule,
    EvolutionJob,
    JobStatus,
)
from loreley.scheduler.job_scheduler import FailedCandidateRepairSampler

log = logger.bind(module="api.repair")

_ACTIVE_JOB_STATUSES = (JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING)


class RepairNotFoundError(RuntimeError):
    """Raised when a repair candidate cannot be found."""


class RepairConflictError(RuntimeError):
    """Raised when an operator action conflicts with active repair work."""


class RepairValidationError(RuntimeError):
    """Raised when a repair request is invalid for current settings."""


@dataclass(frozen=True, slots=True)
class RepairPoolPage:
    items: list[dict[str, object]]
    next_cursor: str | None
    summary: dict[str, object]


def list_repair_pool_page(
    *,
    repair_state: str | None = None,
    lifecycle_status: str | None = None,
    failure_kind: str | None = None,
    campaign_program_hash: str | None = None,
    limit: int = 100,
    cursor: str | None = None,
) -> RepairPoolPage:
    """Return cursor-paginated failed candidates for repair pool inspection."""

    limit, _ = normalize_pagination(limit, 0)
    with session_scope() as session:
        stmt = _repair_pool_stmt(
            repair_state=repair_state,
            lifecycle_status=lifecycle_status,
            failure_kind=failure_kind,
            campaign_program_hash=campaign_program_hash,
        )
        if cursor:
            try:
                payload = decode_cursor(cursor)
                cursor_ts = _normalize_cursor_datetime(payload.get("updated_at"))
                cursor_candidate_id = UUID(str(payload.get("candidate_id")))
            except (PaginationCursorError, ValueError) as exc:
                raise PaginationCursorError("Repair pool cursor is invalid.") from exc
            stmt = stmt.where(
                or_(
                    CandidateCommit.updated_at < cursor_ts,
                    and_(
                        CandidateCommit.updated_at == cursor_ts,
                        CandidateCommit.id < cursor_candidate_id,
                    ),
                )
            )
        stmt = stmt.order_by(CandidateCommit.updated_at.desc(), CandidateCommit.id.desc())
        rows = list(session.execute(stmt.limit(limit + 1)).scalars())
        page_rows = rows[:limit]
        payloads = _candidate_payloads(session=session, rows=page_rows)
        next_cursor = (
            _encode_candidate_cursor(page_rows[-1])
            if len(rows) > limit and page_rows
            else None
        )
        summary = _repair_pool_summary_from_session(session)
    return RepairPoolPage(items=payloads, next_cursor=next_cursor, summary=summary)


def repair_pool_summary() -> dict[str, object]:
    """Return aggregate repair pool counts for operator status."""

    with session_scope() as session:
        return _repair_pool_summary_from_session(session)


def schedule_one_repair(*, settings: Settings | None = None) -> dict[str, object]:
    """Schedule one repair job using the existing scheduler repair sampler."""

    active_settings = settings or get_settings()
    if not bool(active_settings.failed_candidate_repair_enabled):
        return _repair_schedule_noop(
            "Repair scheduling is disabled by FAILED_CANDIDATE_REPAIR_ENABLED.",
        )
    max_per_tick = max(0, int(active_settings.failed_candidate_repair_max_jobs_per_tick))
    if max_per_tick <= 0:
        return _repair_schedule_noop(
            "Repair scheduling is disabled by FAILED_CANDIDATE_REPAIR_MAX_JOBS_PER_TICK.",
        )
    max_active = max(0, int(active_settings.failed_candidate_repair_max_active_jobs))
    if max_active <= 0:
        return _repair_schedule_noop(
            "Repair scheduling is disabled by FAILED_CANDIDATE_REPAIR_MAX_ACTIVE_JOBS.",
        )
    max_tokens = max(0, int(active_settings.failed_candidate_repair_max_tokens))
    if max_tokens <= 0:
        return _repair_schedule_noop(
            "Repair scheduling is disabled by FAILED_CANDIDATE_REPAIR_MAX_TOKENS.",
        )
    try:
        sampler = FailedCandidateRepairSampler(settings=active_settings)

        def _schedule_locked() -> dict[str, object]:
            active_repair_jobs = sampler.count_active_repair_jobs()
            if active_repair_jobs >= max_active:
                return _repair_schedule_noop(
                    "Active repair jobs are already at FAILED_CANDIDATE_REPAIR_MAX_ACTIVE_JOBS.",
                )
            if _manual_repair_tokens_available(settings=active_settings) <= 0:
                return _repair_schedule_noop(
                    "Repair scheduling is blocked by the failed-candidate repair token budget.",
                )
            result = sampler.schedule_one()
            if result is None:
                log.info("Repair schedule-one requested but no eligible candidate was scheduled")
                return _repair_schedule_noop("No eligible repair candidate was scheduled.")
            log.bind(
                job_id=str(result.job_id),
                repair_source_candidate_id=str(result.repair_source_candidate_id),
            ).info("Repair schedule-one created job")
            return {
                "scheduled": True,
                "job_id": result.job_id,
                "repair_source_candidate_id": result.repair_source_candidate_id,
                "base_commit_hash": result.base_commit_hash,
                "message": "Repair job scheduled.",
            }

        return _with_manual_repair_schedule_lock(
            callback=_schedule_locked,
        )
    except ValueError as exc:
        raise RepairValidationError(str(exc)) from exc


def update_candidate_operator_state(*, candidate_id: UUID, action: str) -> dict[str, object]:
    """Quarantine, discard, or restore a repair-pool candidate."""

    action = normalize_single_line(str(action or "")).lower()
    if action not in {"quarantine", "discard", "restore"}:
        raise RepairValidationError(f"Unsupported repair candidate action: {action!r}.")

    with session_scope() as session:
        candidate = _locked_repair_candidate(
            session=session,
            candidate_id=candidate_id,
        )
        if candidate is None:
            raise RepairNotFoundError("Repair candidate not found.")
        if normalize_single_line(str(candidate.evaluation_status or "")).lower() != "candidate_failed":
            raise RepairNotFoundError("Repair candidate not found.")
        active_job = _active_repair_job_for_candidate(session=session, candidate_id=candidate_id)
        if active_job is not None:
            raise RepairConflictError("Candidate has an active repair job.")

        if action == "quarantine":
            candidate.lifecycle_status = "quarantined"
            candidate.repair_state = "quarantined"
        elif action == "discard":
            candidate.lifecycle_status = "discarded"
            candidate.repair_state = "discarded"
        else:
            candidate.lifecycle_status = "active"
            candidate.repair_state = "audit_only"

        session.flush()
        payload = _candidate_payloads(session=session, rows=[candidate])[0]
        log.bind(
            candidate_id=str(candidate_id),
            action=action,
            lifecycle_status=candidate.lifecycle_status,
            repair_state=candidate.repair_state,
        ).info("Repair candidate operator state updated")
        return payload


def _repair_schedule_noop(message: str) -> dict[str, object]:
    log.info("Repair schedule-one did not schedule job: {}", message)
    return {
        "scheduled": False,
        "job_id": None,
        "repair_source_candidate_id": None,
        "base_commit_hash": None,
        "message": message,
    }


def _manual_repair_tokens_available(*, settings: Settings) -> int:
    max_tokens = max(0, int(settings.failed_candidate_repair_max_tokens))
    if max_tokens <= 0:
        return 0
    normal_jobs_per_token = max(1, int(settings.failed_candidate_repair_normal_jobs_per_token))
    with session_scope() as session:
        completed_normal_jobs = int(
            session.execute(
                select(func.count(EvolutionJob.id)).where(
                    EvolutionJob.status == JobStatus.SUCCEEDED,
                    EvolutionJob.job_kind != "repair",
                )
            ).scalar_one()
        )
        scheduled_repair_jobs = int(
            session.execute(
                select(func.count(EvolutionJob.id)).where(
                    EvolutionJob.job_kind == "repair",
                )
            ).scalar_one()
        )
    earned = completed_normal_jobs // normal_jobs_per_token
    return min(max_tokens, max(0, earned - scheduled_repair_jobs))


def _locked_repair_candidate(
    *,
    session: object,
    candidate_id: UUID,
) -> CandidateCommit | None:
    return (
        session.execute(
            select(CandidateCommit)
            .where(CandidateCommit.id == candidate_id)
            .with_for_update()
        ).scalar_one_or_none()
    )


def _repair_pool_stmt(
    *,
    repair_state: str | None,
    lifecycle_status: str | None,
    failure_kind: str | None,
    campaign_program_hash: str | None,
):
    stmt = select(CandidateCommit).where(CandidateCommit.evaluation_status == "candidate_failed")
    for column, raw in (
        (CandidateCommit.repair_state, repair_state),
        (CandidateCommit.lifecycle_status, lifecycle_status),
        (CandidateCommit.failure_kind, failure_kind),
        (CandidateCommit.campaign_program_hash, campaign_program_hash),
    ):
        value = normalize_single_line(str(raw or ""))
        if value:
            stmt = stmt.where(column == value)
    return stmt


def _candidate_payloads(*, session: object, rows: list[CandidateCommit]) -> list[dict[str, object]]:
    if not rows:
        return []
    candidate_ids = [row.id for row in rows]
    active_jobs = _active_repair_jobs_by_candidate(session=session, candidate_ids=candidate_ids)
    last_status_by_job_id = _job_status_by_id(
        session=session,
        job_ids=[row.last_repair_job_id for row in rows if row.last_repair_job_id is not None],
    )
    diagnostics = _diagnostics_by_id(
        session=session,
        diagnostic_ids=[row.failure_evidence_id for row in rows if row.failure_evidence_id is not None],
    )
    payloads: list[dict[str, object]] = []
    for row in rows:
        active_job = active_jobs.get(row.id)
        diagnostic = diagnostics.get(row.failure_evidence_id) if row.failure_evidence_id else None
        payloads.append(
            {
                "id": row.id,
                "commit_hash": row.commit_hash,
                "git_parent_commit_hash": row.git_parent_commit_hash,
                "nearest_viable_ancestor_hash": row.nearest_viable_ancestor_hash,
                "island_id": row.island_id,
                "produced_by_job_id": row.produced_by_job_id,
                "job_kind": row.job_kind,
                "repair_source_candidate_id": row.repair_source_candidate_id,
                "campaign_program_hash": row.campaign_program_hash,
                "publication_status": row.publication_status,
                "evaluation_status": row.evaluation_status,
                "archive_status": row.archive_status,
                "lifecycle_status": row.lifecycle_status,
                "failure_stage": row.failure_stage,
                "failure_kind": row.failure_kind,
                "failure_summary": row.failure_summary,
                "repair_state": row.repair_state,
                "failed_depth": int(row.failed_depth or 0),
                "repair_attempts": int(row.repair_attempts or 0),
                "last_repair_job_id": row.last_repair_job_id,
                "last_repair_job_status": _status_value(last_status_by_job_id.get(row.last_repair_job_id)),
                "active_repair_job_id": getattr(active_job, "id", None),
                "active_repair_job_status": _status_value(getattr(active_job, "status", None)),
                "diagnostic_policy_passed": (
                    bool(getattr(diagnostic, "policy_passed", False))
                    if diagnostic is not None
                    else None
                ),
                "diagnostic_summary": _diagnostic_summary(diagnostic) or row.failure_summary,
                "diagnostic_omitted_reasons": list(getattr(diagnostic, "omitted_reasons", []) or []),
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }
        )
    return payloads


def _active_repair_job_for_candidate(*, session: object, candidate_id: UUID) -> EvolutionJob | None:
    return (
        session.execute(
            select(EvolutionJob)
            .where(
                EvolutionJob.job_kind == "repair",
                EvolutionJob.repair_source_candidate_id == candidate_id,
                EvolutionJob.status.in_(_ACTIVE_JOB_STATUSES),
            )
            .order_by(EvolutionJob.created_at.desc())
            .limit(1)
        )
        .scalars()
        .first()
    )


def _active_repair_jobs_by_candidate(
    *,
    session: object,
    candidate_ids: list[UUID],
) -> dict[UUID, EvolutionJob]:
    if not candidate_ids:
        return {}
    rows = list(
        session.execute(
            select(EvolutionJob)
            .where(
                EvolutionJob.job_kind == "repair",
                EvolutionJob.repair_source_candidate_id.in_(candidate_ids),
                EvolutionJob.status.in_(_ACTIVE_JOB_STATUSES),
            )
            .order_by(EvolutionJob.created_at.desc(), EvolutionJob.id.desc())
        ).scalars()
    )
    result: dict[UUID, EvolutionJob] = {}
    for row in rows:
        source_id = row.repair_source_candidate_id
        if source_id is not None and source_id not in result:
            result[source_id] = row
    return result


def _job_status_by_id(*, session: object, job_ids: list[UUID]) -> dict[UUID, object]:
    if not job_ids:
        return {}
    rows = session.execute(
        select(EvolutionJob.id, EvolutionJob.status).where(EvolutionJob.id.in_(job_ids))
    ).all()
    return {job_id: status for job_id, status in rows}


def _diagnostics_by_id(
    *,
    session: object,
    diagnostic_ids: list[UUID],
) -> dict[UUID, DiagnosticCapsule]:
    if not diagnostic_ids:
        return {}
    rows = list(
        session.execute(
            select(DiagnosticCapsule).where(DiagnosticCapsule.id.in_(diagnostic_ids))
        ).scalars()
    )
    return {row.id: row for row in rows}


def _diagnostic_summary(row: DiagnosticCapsule | None) -> str | None:
    if row is None:
        return None
    payload = getattr(row, "payload", None)
    if not isinstance(payload, dict):
        return None
    for key in (
        "safe_failure_summary",
        "failing_tests_summary",
        "compiler_errors_summary",
        "stack_trace_summary",
        "diff_summary",
    ):
        value = normalize_single_line(str(payload.get(key) or ""))
        if value:
            return clamp_text(value, 512)
    return None


def _repair_pool_summary_from_session(session: object) -> dict[str, object]:
    base = CandidateCommit.evaluation_status == "candidate_failed"
    total = int(
        session.execute(
            select(func.count(CandidateCommit.id)).where(base)
        ).scalar_one()
        or 0
    )
    active_repair_jobs = int(
        session.execute(
            select(func.count(EvolutionJob.id)).where(
                EvolutionJob.job_kind == "repair",
                EvolutionJob.status.in_(_ACTIVE_JOB_STATUSES),
            )
        ).scalar_one()
        or 0
    )
    return {
        "total_failed_candidates": total,
        "active_repair_jobs": active_repair_jobs,
        "by_repair_state": _group_counts(
            session=session,
            column=CandidateCommit.repair_state,
            where_clause=base,
        ),
        "by_lifecycle_status": _group_counts(
            session=session,
            column=CandidateCommit.lifecycle_status,
            where_clause=base,
        ),
        "by_failure_kind": _group_counts(
            session=session,
            column=CandidateCommit.failure_kind,
            where_clause=base,
            default="unknown",
        ),
    }


def _group_counts(
    *,
    session: object,
    column: object,
    where_clause: object,
    default: str = "",
) -> dict[str, int]:
    rows = session.execute(
        select(column, func.count()).where(where_clause).group_by(column)
    ).all()
    counts: dict[str, int] = {}
    for value, count in rows:
        key = normalize_single_line(str(value or default or "unknown")) or "unknown"
        counts[key] = int(count or 0)
    return counts


def _normalize_cursor_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise PaginationCursorError("Repair pool cursor is missing updated_at.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise PaginationCursorError("Repair pool cursor has an invalid timestamp.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _encode_candidate_cursor(candidate: CandidateCommit) -> str:
    if not isinstance(candidate.updated_at, datetime):
        raise ValueError("repair pool cursor requires updated_at")
    return encode_cursor(
        {
            "updated_at": candidate.updated_at.isoformat(),
            "candidate_id": str(candidate.id),
        }
    )


def _status_value(value: object) -> str | None:
    status_value = getattr(value, "value", value)
    normalized = normalize_single_line(str(status_value or ""))
    return normalized or None

"""Job queries for the UI API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.orm import aliased

from loreley.api.pagination import PaginationCursorError, decode_cursor, encode_cursor, normalize_pagination
from loreley.config import Settings, get_settings
from loreley.core.candidate_fate import CANDIDATE_FATE_LABELS
from loreley.core.job_retry import (
    db_utc_now,
    job_retry_state,
    retry_failed_stale_jobs_payload,
    retry_job_row,
)
from loreley.db.base import session_scope
from loreley.db.models import (
    CandidateCommit,
    EvaluationArtifactRecord,
    EvolutionJob,
    JobArtifacts,
    JobStatus,
    MapElitesArchiveCell,
)


@dataclass(frozen=True, slots=True)
class JobPage:
    items: list[EvolutionJob]
    next_cursor: str | None


class JobNotFoundError(RuntimeError):
    """Raised when a job cannot be found."""


class JobRetryConflictError(RuntimeError):
    """Raised when a job is not retryable."""


class JobRetryValidationError(RuntimeError):
    """Raised when a bulk retry request is invalid."""


_EVIDENCE_HAS_EVIDENCE = "has_evidence"
_EVIDENCE_AGENT_VISIBLE = "agent_visible"
_EVIDENCE_NONE = "none"
JOB_EVIDENCE_FILTERS = frozenset(
    {
        _EVIDENCE_HAS_EVIDENCE,
        _EVIDENCE_AGENT_VISIBLE,
        _EVIDENCE_NONE,
    }
)
_DISCARDED_STATES = {"discarded", "quarantined"}
_REPAIR_PENDING_STATES = {"eligible", "scheduled", "repairing"}
_NON_CANDIDATE_FAILURE_OUTCOMES = {
    "evaluator_failed",
    "infrastructure_failed",
    "inconclusive",
}


def _normalize_cursor_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise PaginationCursorError("Jobs cursor is missing sort_ts.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise PaginationCursorError("Jobs cursor has an invalid timestamp.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _encode_job_cursor(job: EvolutionJob) -> str:
    sort_ts = getattr(job, "completed_at", None) or getattr(job, "created_at", None)
    if not isinstance(sort_ts, datetime):
        raise ValueError("jobs cursor requires a timestamp")
    return encode_cursor(
        {
            "sort_ts": sort_ts.isoformat(),
            "job_id": str(job.id),
        }
    )


def _normalize_candidate_fate_filter(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    if normalized not in CANDIDATE_FATE_LABELS:
        raise ValueError(f"Unsupported candidate_fate filter: {normalized}")
    return normalized


def _normalize_evidence_filter(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    if normalized not in JOB_EVIDENCE_FILTERS:
        raise ValueError(f"Unsupported evidence filter: {normalized}")
    return normalized


def _projection_filters(
    *,
    candidate_fate: str | None,
    evidence: str | None,
) -> tuple[str | None, str | None]:
    return (
        _normalize_candidate_fate_filter(candidate_fate),
        _normalize_evidence_filter(evidence),
    )


def _base_jobs_stmt(*, status: JobStatus | None, job_kind: str | None):
    stmt = select(EvolutionJob)
    if status is not None:
        stmt = stmt.where(EvolutionJob.status == status)
    job_kind_filter = str(job_kind or "").strip()
    if job_kind_filter:
        stmt = stmt.where(EvolutionJob.job_kind == job_kind_filter)
    return stmt


def _apply_projection_filter_predicates(
    stmt,
    *,
    candidate_fate: str | None,
    evidence: str | None,
):
    if candidate_fate is not None:
        stmt = _apply_candidate_fate_filter(stmt, candidate_fate=candidate_fate)
    if evidence is not None:
        stmt = _apply_evidence_filter(stmt, evidence=evidence)
    return stmt


def _apply_evidence_filter(stmt, *, evidence: str):
    has_evidence = _evaluation_artifact_exists(
        _job_candidate_commit_hash_expr(EvolutionJob),
        EvaluationArtifactRecord.visibility != "hidden",
    )
    if evidence == _EVIDENCE_HAS_EVIDENCE:
        return stmt.where(has_evidence)
    if evidence == _EVIDENCE_NONE:
        return stmt.where(~has_evidence)
    if evidence == _EVIDENCE_AGENT_VISIBLE:
        return stmt.where(
            _evaluation_artifact_exists(
                _job_candidate_commit_hash_expr(EvolutionJob),
                EvaluationArtifactRecord.visibility == "agent_visible",
            )
        )
    return stmt


def _evaluation_artifact_exists(commit_hash_expr, *criteria):
    return (
        select(EvaluationArtifactRecord.job_id)
        .where(
            EvaluationArtifactRecord.commit_hash == commit_hash_expr,
            *criteria,
        )
        .exists()
    )


def _apply_candidate_fate_filter(stmt, *, candidate_fate: str):
    candidate = aliased(CandidateCommit)
    producer_job = aliased(EvolutionJob)
    commit_hash = _job_candidate_commit_hash_expr(EvolutionJob)
    stmt = stmt.outerjoin(candidate, candidate.commit_hash == commit_hash)
    stmt = stmt.outerjoin(producer_job, producer_job.id == candidate.produced_by_job_id)
    return stmt.where(
        _candidate_fate_label_expr(candidate=candidate, producer_job=producer_job)
        == candidate_fate
    )


def _candidate_fate_label_expr(
    *,
    candidate,
    producer_job,
):
    producer_applies = and_(
        candidate.produced_by_job_id.is_not(None),
        candidate.produced_by_job_id != EvolutionJob.id,
        producer_job.id.is_not(None),
    )
    effective_status = _effective_job_column(
        producer_applies=producer_applies,
        producer_job=producer_job,
        name="status",
    )
    effective_ingestion_status = _lower_text_expr(
        _effective_job_column(
            producer_applies=producer_applies,
            producer_job=producer_job,
            name="ingestion_status",
        )
    )
    effective_ingestion_code = _effective_job_column(
        producer_applies=producer_applies,
        producer_job=producer_job,
        name="ingestion_status_code",
    )
    effective_job_kind_commit_hash = _job_candidate_commit_hash_expr(
        producer_job,
        fallback_job=EvolutionJob,
        producer_applies=producer_applies,
    )
    fate_commit_hash = func.coalesce(
        func.nullif(candidate.commit_hash, ""),
        func.nullif(effective_job_kind_commit_hash, ""),
        "",
    )
    fate_island_id = func.coalesce(
        func.nullif(
            _effective_job_column(
                producer_applies=producer_applies,
                producer_job=producer_job,
                name="island_id",
            ),
            "",
        ),
        func.nullif(candidate.island_id, ""),
        func.nullif(EvolutionJob.island_id, ""),
        "",
    )
    current_archive_member = and_(
        fate_commit_hash != "",
        fate_island_id != "",
        select(MapElitesArchiveCell.cell_index)
        .where(
            MapElitesArchiveCell.commit_hash == fate_commit_hash,
            MapElitesArchiveCell.island_id == fate_island_id,
        )
        .exists(),
    )

    candidate_evaluation_status = _lower_text_expr(candidate.evaluation_status)
    candidate_archive_status = _lower_text_expr(candidate.archive_status)
    candidate_repair_state = _lower_text_expr(candidate.repair_state)
    candidate_lifecycle_status = _lower_text_expr(candidate.lifecycle_status)
    candidate_failure_stage = _lower_text_expr(candidate.failure_stage)
    candidate_passed = or_(
        candidate_evaluation_status == "passed",
        and_(effective_status == JobStatus.SUCCEEDED, fate_commit_hash != ""),
    )

    return case(
        (
            or_(
                candidate_lifecycle_status.in_(_DISCARDED_STATES),
                candidate_repair_state.in_(_DISCARDED_STATES),
            ),
            "discarded_for_sampling",
        ),
        (candidate_repair_state.in_(_REPAIR_PENDING_STATES), "repair_pending"),
        (candidate_failure_stage == "policy", "policy_failed"),
        (candidate_evaluation_status == "candidate_failed", "candidate_failed"),
        (candidate_evaluation_status.in_(_NON_CANDIDATE_FAILURE_OUTCOMES), "unknown"),
        (
            and_(
                effective_ingestion_status == "succeeded",
                effective_ingestion_code.is_not(None),
                effective_ingestion_code > 0,
                effective_ingestion_code == 2,
            ),
            "elite_inserted",
        ),
        (
            and_(
                effective_ingestion_status == "succeeded",
                effective_ingestion_code.is_not(None),
                effective_ingestion_code > 0,
            ),
            "elite_replaced",
        ),
        (current_archive_member, "elite_retained"),
        (~candidate_passed, "unknown"),
        (
            or_(
                candidate_archive_status == "rejected",
                effective_ingestion_status == "skipped",
            ),
            "valid_not_elite",
        ),
        (candidate_archive_status == "member", "valid_not_elite"),
        (effective_ingestion_status == "failed", "valid_not_considered"),
        else_="valid_not_considered",
    )


def _effective_job_column(
    *,
    producer_applies,
    producer_job,
    name: str,
):
    return case(
        (producer_applies, getattr(producer_job, name)),
        else_=getattr(EvolutionJob, name),
    )


def _lower_text_expr(expr):
    return func.lower(func.coalesce(expr, ""))


def _job_candidate_commit_hash_expr(
    job_model,
    *,
    fallback_job=None,
    producer_applies=None,
):
    primary = func.coalesce(
        func.nullif(job_model.result_commit_hash, ""),
        func.nullif(job_model.candidate_commit_hash, ""),
        "",
    )
    if fallback_job is None or producer_applies is None:
        return primary
    fallback = func.coalesce(
        func.nullif(fallback_job.result_commit_hash, ""),
        func.nullif(fallback_job.candidate_commit_hash, ""),
        "",
    )
    return case((producer_applies, primary), else_=fallback)


def list_jobs_page(
    *,
    status: JobStatus | None = None,
    job_kind: str | None = None,
    candidate_fate: str | None = None,
    evidence: str | None = None,
    limit: int = 200,
    cursor: str | None = None,
) -> JobPage:
    """Return a cursor-paginated page of jobs ordered newest-first."""

    limit, _ = normalize_pagination(limit, 0)
    candidate_fate, evidence = _projection_filters(
        candidate_fate=candidate_fate,
        evidence=evidence,
    )
    sort_ts = func.coalesce(EvolutionJob.completed_at, EvolutionJob.created_at)

    with session_scope() as session:
        stmt = _base_jobs_stmt(status=status, job_kind=job_kind)
        stmt = _apply_projection_filter_predicates(
            stmt,
            candidate_fate=candidate_fate,
            evidence=evidence,
        )
        if cursor:
            try:
                payload = decode_cursor(cursor)
                cursor_ts = _normalize_cursor_datetime(payload.get("sort_ts"))
                cursor_job_id = UUID(str(payload.get("job_id")))
            except (PaginationCursorError, ValueError) as exc:
                raise PaginationCursorError("Jobs cursor is invalid.") from exc
            stmt = stmt.where(
                or_(
                    sort_ts < cursor_ts,
                    and_(
                        sort_ts == cursor_ts,
                        EvolutionJob.id < cursor_job_id,
                    ),
                )
            )
        stmt = stmt.order_by(sort_ts.desc(), EvolutionJob.id.desc())
        stmt = stmt.limit(limit + 1)
        rows = list(session.execute(stmt).scalars())

    items = rows[:limit]
    next_cursor = _encode_job_cursor(items[-1]) if len(rows) > limit and items else None
    return JobPage(items=items, next_cursor=next_cursor)


def list_jobs(
    *,
    status: JobStatus | None = None,
    job_kind: str | None = None,
    candidate_fate: str | None = None,
    evidence: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[EvolutionJob]:
    """Return jobs ordered by completion time (or creation time) descending."""

    limit, offset = normalize_pagination(limit, offset)
    candidate_fate, evidence = _projection_filters(
        candidate_fate=candidate_fate,
        evidence=evidence,
    )

    with session_scope() as session:
        stmt = _base_jobs_stmt(status=status, job_kind=job_kind)
        stmt = _apply_projection_filter_predicates(
            stmt,
            candidate_fate=candidate_fate,
            evidence=evidence,
        )
        stmt = stmt.order_by(
            EvolutionJob.completed_at.desc().nullslast(),
            EvolutionJob.created_at.desc(),
            EvolutionJob.id.desc(),
        )
        stmt = stmt.limit(limit).offset(offset)
        return list(session.execute(stmt).scalars())


def get_job(*, job_id: UUID) -> EvolutionJob | None:
    """Return a single job or None."""

    with session_scope() as session:
        return session.get(EvolutionJob, job_id)


def get_job_artifacts(*, job_id: UUID) -> JobArtifacts | None:
    """Return JobArtifacts row for a job."""

    with session_scope() as session:
        return session.get(JobArtifacts, job_id)


def retry_job_by_id(
    *,
    job_id: UUID,
    reason: str | None = None,
) -> dict[str, object]:
    """Retry one FAILED or stale/missing-lease RUNNING job."""

    with session_scope() as session:
        job = session.get(EvolutionJob, job_id)
        if job is None:
            raise JobNotFoundError("Job not found.")
        now = db_utc_now(session)
        retryable, lease_state = job_retry_state(job=job, now=now)
        if not retryable:
            status = getattr(getattr(job, "status", None), "value", getattr(job, "status", None))
            raise JobRetryConflictError(
                f"Only failed or stuck RUNNING jobs can be retried "
                f"(status={status}, lease_state={lease_state or 'n/a'})."
            )
        return retry_job_row(
            job=job,
            reason=str(reason or "").strip() or "manual retry requested via UI API",
            now=now,
        )


def retry_failed_stale_jobs(
    *,
    retry_all: bool,
    limit: int | None,
    reason: str | None = None,
    settings: Settings | None = None,
) -> dict[str, object]:
    """Retry FAILED jobs that exhausted stale-lease recovery."""

    if bool(retry_all) and limit is not None:
        raise JobRetryValidationError("Use either all=true or limit, not both.")
    if not bool(retry_all) and limit is None:
        raise JobRetryValidationError("Use either all=true or limit.")
    active_settings = settings or get_settings()
    with session_scope() as session:
        now = db_utc_now(session)
        return retry_failed_stale_jobs_payload(
            session=session,
            max_recovery_attempts=int(active_settings.scheduler_stale_running_max_recovery_attempts),
            retry_all=bool(retry_all),
            limit=limit,
            reason=str(reason or "").strip() or "manual retry requested via UI API",
            now=now,
        )

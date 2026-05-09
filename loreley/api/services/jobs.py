"""Job queries for the UI API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import and_, func, or_, select

from loreley.api.pagination import PaginationCursorError, decode_cursor, encode_cursor, normalize_pagination
from loreley.api.services.candidate_fates import job_candidate_commit_hash, load_candidate_fates_for_jobs
from loreley.api.services.evidence import load_evidence_indicators_by_commit_hash
from loreley.config import Settings, get_settings
from loreley.core.candidate_fate import CANDIDATE_FATE_LABELS
from loreley.core.job_retry import (
    db_utc_now,
    job_retry_state,
    retry_failed_stale_jobs_payload,
    retry_job_row,
)
from loreley.db.base import session_scope
from loreley.db.models import EvolutionJob, JobArtifacts, JobStatus


@dataclass(frozen=True, slots=True)
class JobPage:
    items: list[EvolutionJob]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class _ProjectedJobPageScan:
    status: JobStatus | None
    job_kind: str | None
    candidate_fate: str | None
    evidence: str | None
    limit: int
    scan_limit: int
    sort_ts: object


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
_FILTER_SCAN_BATCH_MIN = 100


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


def _job_sort_datetime(job: EvolutionJob) -> datetime:
    sort_ts = getattr(job, "completed_at", None) or getattr(job, "created_at", None)
    if not isinstance(sort_ts, datetime):
        raise ValueError("jobs cursor requires a timestamp")
    return sort_ts


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


def _filtered_jobs(
    rows: list[EvolutionJob],
    *,
    candidate_fate: str | None,
    evidence: str | None,
) -> list[EvolutionJob]:
    if not rows or (candidate_fate is None and evidence is None):
        return rows

    fates = load_candidate_fates_for_jobs(rows) if candidate_fate is not None else {}
    indicators = (
        load_evidence_indicators_by_commit_hash(
            [job_candidate_commit_hash(row) for row in rows]
        )
        if evidence is not None
        else {}
    )
    return [
        row
        for row in rows
        if _job_matches_projection_filters(
            row,
            candidate_fate=candidate_fate,
            evidence=evidence,
            fates=fates,
            indicators=indicators,
        )
    ]


def _job_matches_projection_filters(
    row: EvolutionJob,
    *,
    candidate_fate: str | None,
    evidence: str | None,
    fates: dict[str, object],
    indicators: dict[str, object],
) -> bool:
    return _job_matches_fate_filter(row, candidate_fate, fates) and _job_matches_evidence_filter(
        row,
        evidence,
        indicators,
    )


def _job_matches_fate_filter(
    row: EvolutionJob,
    candidate_fate: str | None,
    fates: dict[str, object],
) -> bool:
    if candidate_fate is None:
        return True
    fate = fates.get(str(getattr(row, "id", "") or ""))
    label = str(getattr(fate, "label", None) or "unknown").strip() or "unknown"
    return label == candidate_fate


def _job_matches_evidence_filter(
    row: EvolutionJob,
    evidence: str | None,
    indicators: dict[str, object],
) -> bool:
    if evidence is None:
        return True
    indicator = indicators.get(job_candidate_commit_hash(row))
    has_evidence = bool(getattr(indicator, "has_evaluation_evidence", False))
    agent_visible = int(getattr(indicator, "agent_visible_evidence_count", 0) or 0) > 0
    return _evidence_state_matches_filter(
        evidence=evidence,
        has_evidence=has_evidence,
        agent_visible=agent_visible,
    )


def _evidence_state_matches_filter(
    *,
    evidence: str,
    has_evidence: bool,
    agent_visible: bool,
) -> bool:
    if evidence == _EVIDENCE_HAS_EVIDENCE:
        return has_evidence
    if evidence == _EVIDENCE_AGENT_VISIBLE:
        return agent_visible
    if evidence == _EVIDENCE_NONE:
        return not has_evidence
    return True


def _base_jobs_stmt(*, status: JobStatus | None, job_kind: str | None):
    stmt = select(EvolutionJob)
    if status is not None:
        stmt = stmt.where(EvolutionJob.status == status)
    job_kind_filter = str(job_kind or "").strip()
    if job_kind_filter:
        stmt = stmt.where(EvolutionJob.job_kind == job_kind_filter)
    return stmt


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

    if candidate_fate is not None or evidence is not None:
        scan = _ProjectedJobPageScan(
            status=status,
            job_kind=job_kind,
            candidate_fate=candidate_fate,
            evidence=evidence,
            limit=limit,
            scan_limit=max(limit + 1, _FILTER_SCAN_BATCH_MIN),
            sort_ts=sort_ts,
        )
        return _list_jobs_page_with_projection_filters(
            scan=scan,
            cursor=cursor,
        )

    with session_scope() as session:
        stmt = _base_jobs_stmt(status=status, job_kind=job_kind)
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


def _list_jobs_page_with_projection_filters(
    *,
    scan: _ProjectedJobPageScan,
    cursor: str | None,
) -> JobPage:
    scan_after = _decode_job_cursor(cursor)
    items = _collect_projected_page_items(scan=scan, scan_after=scan_after)
    page_items = items[:scan.limit]
    next_cursor = _encode_job_cursor(page_items[-1]) if len(items) > scan.limit and page_items else None
    return JobPage(items=page_items, next_cursor=next_cursor)


def _decode_job_cursor(cursor: str | None) -> tuple[datetime, UUID] | None:
    if not cursor:
        return None
    try:
        payload = decode_cursor(cursor)
        return (
            _normalize_cursor_datetime(payload.get("sort_ts")),
            UUID(str(payload.get("job_id"))),
        )
    except (PaginationCursorError, ValueError) as exc:
        raise PaginationCursorError("Jobs cursor is invalid.") from exc


def _collect_projected_page_items(
    *,
    scan: _ProjectedJobPageScan,
    scan_after: tuple[datetime, UUID] | None,
) -> list[EvolutionJob]:
    items: list[EvolutionJob] = []
    with session_scope() as session:
        while len(items) <= scan.limit:
            stmt = _projected_page_stmt(scan=scan, scan_after=scan_after)
            rows = list(session.execute(stmt).scalars())
            if not rows:
                break
            items.extend(
                _filtered_jobs(
                    rows,
                    candidate_fate=scan.candidate_fate,
                    evidence=scan.evidence,
                )
            )
            scan_after = _job_cursor_tuple(rows[-1])
            if len(rows) < scan.scan_limit:
                break
    return items


def _projected_page_stmt(
    *,
    scan: _ProjectedJobPageScan,
    scan_after: tuple[datetime, UUID] | None,
):
    stmt = _base_jobs_stmt(status=scan.status, job_kind=scan.job_kind)
    if scan_after is not None:
        cursor_ts, cursor_job_id = scan_after
        stmt = stmt.where(
            or_(
                scan.sort_ts < cursor_ts,
                and_(
                    scan.sort_ts == cursor_ts,
                    EvolutionJob.id < cursor_job_id,
                ),
            )
        )
    return stmt.order_by(scan.sort_ts.desc(), EvolutionJob.id.desc()).limit(scan.scan_limit)


def _job_cursor_tuple(job: EvolutionJob) -> tuple[datetime, UUID]:
    return (_job_sort_datetime(job), job.id)


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

    if candidate_fate is not None or evidence is not None:
        return _list_jobs_with_projection_filters(
            status=status,
            job_kind=job_kind,
            candidate_fate=candidate_fate,
            evidence=evidence,
            limit=limit,
            offset=offset,
        )

    with session_scope() as session:
        stmt = _base_jobs_stmt(status=status, job_kind=job_kind)
        stmt = stmt.order_by(
            EvolutionJob.completed_at.desc().nullslast(),
            EvolutionJob.created_at.desc(),
            EvolutionJob.id.desc(),
        )
        stmt = stmt.limit(limit).offset(offset)
        return list(session.execute(stmt).scalars())


def _list_jobs_with_projection_filters(
    *,
    status: JobStatus | None,
    job_kind: str | None,
    candidate_fate: str | None,
    evidence: str | None,
    limit: int,
    offset: int,
) -> list[EvolutionJob]:
    selected: list[EvolutionJob] = []
    skipped = 0
    raw_offset = 0
    scan_limit = max(limit, _FILTER_SCAN_BATCH_MIN)
    with session_scope() as session:
        while len(selected) < limit:
            rows = _query_projected_offset_batch(
                session=session,
                status=status,
                job_kind=job_kind,
                scan_limit=scan_limit,
                raw_offset=raw_offset,
            )
            if not rows:
                break
            filtered = _filtered_jobs(
                rows,
                candidate_fate=candidate_fate,
                evidence=evidence,
            )
            skipped = _append_projected_offset_matches(
                selected=selected,
                filtered=filtered,
                skipped=skipped,
                offset=offset,
                limit=limit,
            )
            raw_offset += len(rows)
            if len(rows) < scan_limit:
                break
    return selected


def _query_projected_offset_batch(
    *,
    session: object,
    status: JobStatus | None,
    job_kind: str | None,
    scan_limit: int,
    raw_offset: int,
) -> list[EvolutionJob]:
    stmt = _base_jobs_stmt(status=status, job_kind=job_kind)
    stmt = stmt.order_by(
        EvolutionJob.completed_at.desc().nullslast(),
        EvolutionJob.created_at.desc(),
        EvolutionJob.id.desc(),
    )
    stmt = stmt.limit(scan_limit).offset(raw_offset)
    return list(session.execute(stmt).scalars())


def _append_projected_offset_matches(
    *,
    selected: list[EvolutionJob],
    filtered: list[EvolutionJob],
    skipped: int,
    offset: int,
    limit: int,
) -> int:
    for row in filtered:
        if skipped < offset:
            skipped += 1
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return skipped


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

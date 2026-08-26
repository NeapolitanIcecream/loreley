"""Deterministic, sanitized export of Loreley's evolution timeline."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from loreley.core.evolution_events import (
    ARCHIVE_MEMBER_ADMITTED,
    ARCHIVE_MEMBER_INITIAL_STATE,
    ARCHIVE_MEMBER_MOVED,
    ARCHIVE_MEMBER_REMOVED,
    CODING_FINISHED,
    CODING_STARTED,
    EVALUATION_INVOCATION_STARTED,
    INGESTION_FINISHED,
    INGESTION_STARTED,
    JOB_CANCELLED,
    JOB_FAILED,
    JOB_RECLAIMED,
    JOB_RECOVERY_EXHAUSTED,
    JOB_REQUEUED,
    JOB_RUN_STARTED,
    JOB_SUCCEEDED,
    PLANNING_FINISHED,
    PLANNING_STARTED,
    TIMELINE_HISTORY_BOUNDARY,
)
from loreley.db.base import INSTANCE_SCHEMA_VERSION
from loreley.db.models import (
    CandidateCommit,
    EvaluationAttempt,
    EvaluationResourceLease,
    EvolutionEvent,
    EvolutionJob,
)

TIMELINE_SCHEMA_VERSION = 1
_TERMINAL_EVENT_TYPES = {
    JOB_SUCCEEDED,
    JOB_FAILED,
    JOB_CANCELLED,
    JOB_RECOVERY_EXHAUSTED,
}
_RUN_INTERRUPTION_EVENT_TYPES = {
    JOB_FAILED,
    JOB_RECLAIMED,
    JOB_RECOVERY_EXHAUSTED,
    JOB_REQUEUED,
}
_STAGE_TYPES = {
    "planning": (PLANNING_STARTED, PLANNING_FINISHED),
    "coding": (CODING_STARTED, CODING_FINISHED),
    "evaluation": (
        EVALUATION_INVOCATION_STARTED,
        "evaluation.invocation.finished",
    ),
    "ingestion": (INGESTION_STARTED, INGESTION_FINISHED),
}


@dataclass(frozen=True, slots=True)
class TimelineIssue:
    code: str
    message: str
    job_id: str | None = None
    event_id: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "job_id": self.job_id,
            "event_id": self.event_id,
        }


class TimelineCompletenessError(RuntimeError):
    """Raised when strict export finds incomplete or impossible evidence."""

    def __init__(self, issues: Sequence[TimelineIssue]) -> None:
        self.issues = tuple(issues)
        super().__init__(
            "Evolution timeline failed strict completeness checks: "
            + "; ".join(issue.message for issue in self.issues[:8])
        )


@dataclass(frozen=True, slots=True)
class EvolutionTimelineExport:
    metadata: Mapping[str, object]
    events: tuple[Mapping[str, object], ...]
    issues: tuple[TimelineIssue, ...]

    def records(self) -> tuple[Mapping[str, object], ...]:
        return (self.metadata, *self.events)

    def to_jsonl(self) -> str:
        return "".join(
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for record in self.records()
        )


def export_evolution_timeline(
    session: Session,
    *,
    strict: bool = False,
    job_ids: Sequence[UUID] | None = None,
) -> EvolutionTimelineExport:
    """Build a stable JSONL-ready timeline from authoritative product rows."""

    selected_jobs = _load_jobs(session, job_ids=job_ids)
    selected_job_ids = {job.id for job in selected_jobs}
    ledger_events = _load_ledger_events(
        session,
        selected_job_ids=selected_job_ids,
        filter_jobs=job_ids is not None,
    )
    records: list[dict[str, object]] = [
        _ledger_record(event) for event in ledger_events
    ]
    records.extend(_project_parent_selection(job) for job in selected_jobs)
    records.extend(
        _evaluation_attempt_records(
            session,
            selected_job_ids=selected_job_ids,
            filter_jobs=job_ids is not None,
            ledger_events=ledger_events,
        )
    )
    records.extend(
        _evaluation_lease_records(
            session,
            selected_job_ids=selected_job_ids,
            filter_jobs=job_ids is not None,
        )
    )
    records.extend(_derive_interrupted_stages(records))
    ordered = tuple(sorted(records, key=_timeline_sort_key))
    issues = tuple(_timeline_issues(ordered, jobs=selected_jobs))
    if strict and issues:
        raise TimelineCompletenessError(issues)
    metadata: dict[str, object] = {
        "record_type": "metadata",
        "timeline_schema_version": TIMELINE_SCHEMA_VERSION,
        "database_schema_version": INSTANCE_SCHEMA_VERSION,
        "ordering": ["occurred_at", "event_id"],
        "strict_valid": not issues,
        "issue_count": len(issues),
    }
    return EvolutionTimelineExport(
        metadata=metadata,
        events=ordered,
        issues=issues,
    )


def _load_jobs(
    session: Session,
    *,
    job_ids: Sequence[UUID] | None,
) -> list[EvolutionJob]:
    statement = select(EvolutionJob).order_by(
        EvolutionJob.created_at.asc(),
        EvolutionJob.id.asc(),
    )
    if job_ids is not None:
        ids = tuple(dict.fromkeys(job_ids))
        if not ids:
            return []
        statement = statement.where(EvolutionJob.id.in_(ids))
    return list(session.execute(statement).scalars().all())


def _load_ledger_events(
    session: Session,
    *,
    selected_job_ids: set[UUID],
    filter_jobs: bool,
) -> list[EvolutionEvent]:
    statement = select(EvolutionEvent).order_by(
        EvolutionEvent.occurred_at.asc(),
        EvolutionEvent.id.asc(),
    )
    if filter_jobs:
        statement = statement.where(
            (EvolutionEvent.job_id.in_(tuple(selected_job_ids)))
            | EvolutionEvent.job_id.is_(None)
        )
    return list(session.execute(statement).scalars().all())


def _ledger_record(event: EvolutionEvent) -> dict[str, object]:
    return _event_record(
        event_id=f"evolution_event:{event.id}",
        source="evolution_event",
        event_type=event.event_type,
        occurred_at=event.occurred_at,
        job_id=event.job_id,
        run_token=event.run_token,
        island_id=event.island_id,
        commit_hash=event.commit_hash,
        ordinal=event.ordinal,
        duration_seconds=event.duration_seconds,
        payload=dict(event.payload or {}),
    )


def _project_parent_selection(job: EvolutionJob) -> dict[str, object]:
    return _event_record(
        event_id=f"evolution_job:{job.id}:parent_selected",
        source="evolution_job",
        event_type="parent.selected",
        occurred_at=job.created_at,
        job_id=job.id,
        run_token=None,
        island_id=job.island_id,
        commit_hash=job.base_commit_hash,
        ordinal=job.sampling_ordinal,
        duration_seconds=None,
        payload={
            "base_commit_hash": job.base_commit_hash,
            "inspiration_commit_hashes": list(job.inspiration_commit_hashes or ()),
            "migration_source_island_id": job.migration_source_island_id,
            "migration_commit_hash": job.migration_commit_hash,
            "job_kind": str(job.job_kind or "evolution"),
        },
    )


def _evaluation_attempt_records(
    session: Session,
    *,
    selected_job_ids: set[UUID],
    filter_jobs: bool,
    ledger_events: Sequence[EvolutionEvent],
) -> list[dict[str, object]]:
    statement = (
        select(EvaluationAttempt, CandidateCommit.commit_hash)
        .outerjoin(
            CandidateCommit,
            CandidateCommit.id == EvaluationAttempt.candidate_commit_id,
        )
        .order_by(
            EvaluationAttempt.started_at.asc().nullsfirst(),
            EvaluationAttempt.id.asc(),
        )
    )
    if filter_jobs:
        statement = statement.where(
            EvaluationAttempt.job_id.in_(tuple(selected_job_ids))
        )
    invocation_by_key = {
        (
            event.job_id,
            event.run_token,
            int(event.ordinal or 0),
        ): event
        for event in ledger_events
        if event.event_type == EVALUATION_INVOCATION_STARTED
    }
    records: list[dict[str, object]] = []
    for attempt, candidate_commit_hash in session.execute(statement).all():
        common_payload = {
            "attempt_id": str(attempt.id),
            "protocol": attempt.protocol,
            "evaluator_name": attempt.evaluator_name,
            "evaluator_version": attempt.evaluator_version,
            "measurement_reused": bool(attempt.measurement_reused),
            "reuse_kind": attempt.reuse_kind,
        }
        if attempt.started_at is not None:
            records.append(
                _event_record(
                    event_id=f"evaluation_attempt:{attempt.id}:started",
                    source="evaluation_attempt",
                    event_type="evaluation.attempt.started",
                    occurred_at=attempt.started_at,
                    job_id=attempt.job_id,
                    run_token=attempt.run_token,
                    island_id=None,
                    commit_hash=candidate_commit_hash,
                    ordinal=attempt.attempt_ordinal,
                    duration_seconds=None,
                    payload=common_payload,
                )
            )
        if attempt.finished_at is None:
            continue
        duration = _elapsed_seconds(attempt.started_at, attempt.finished_at)
        finish_payload = {
            **common_payload,
            "outcome": attempt.outcome_kind,
            "failure_stage": attempt.failure_stage,
            "failure_kind": attempt.failure_kind,
        }
        records.append(
            _event_record(
                event_id=f"evaluation_attempt:{attempt.id}:finished",
                source="evaluation_attempt",
                event_type="evaluation.attempt.finished",
                occurred_at=attempt.finished_at,
                job_id=attempt.job_id,
                run_token=attempt.run_token,
                island_id=None,
                commit_hash=candidate_commit_hash,
                ordinal=attempt.attempt_ordinal,
                duration_seconds=duration,
                payload=finish_payload,
            )
        )
        invocation_key = (
            attempt.job_id,
            attempt.run_token,
            int(attempt.attempt_ordinal or 0),
        )
        invocation = invocation_by_key.get(invocation_key)
        if invocation is not None:
            records.append(
                _event_record(
                    event_id=f"evaluation_attempt:{attempt.id}:invocation_finished",
                    source="evaluation_attempt",
                    event_type="evaluation.invocation.finished",
                    occurred_at=attempt.finished_at,
                    job_id=attempt.job_id,
                    run_token=attempt.run_token,
                    island_id=None,
                    commit_hash=candidate_commit_hash,
                    ordinal=attempt.attempt_ordinal,
                    duration_seconds=_elapsed_seconds(
                        invocation.occurred_at,
                        attempt.finished_at,
                    ),
                    payload={
                        "attempt_id": str(attempt.id),
                        "outcome": attempt.outcome_kind,
                        "failure_kind": attempt.failure_kind,
                    },
                )
            )
    return records


def _evaluation_lease_records(
    session: Session,
    *,
    selected_job_ids: set[UUID],
    filter_jobs: bool,
) -> list[dict[str, object]]:
    statement = select(EvaluationResourceLease).order_by(
        EvaluationResourceLease.requested_at.asc(),
        EvaluationResourceLease.id.asc(),
    )
    if filter_jobs:
        statement = statement.where(
            EvaluationResourceLease.job_id.in_(tuple(selected_job_ids))
        )
    records: list[dict[str, object]] = []
    for lease in session.execute(statement).scalars().all():
        payload = {
            "lease_id": str(lease.id),
            "resource_kind": lease.resource_kind,
            "resource_key": lease.resource_key,
            "contract_key": lease.contract_key,
            "slot_index": lease.slot_index,
            "status": lease.status,
        }
        records.append(
            _event_record(
                event_id=f"evaluation_lease:{lease.id}:requested",
                source="evaluation_resource_lease",
                event_type="evaluation.lease.requested",
                occurred_at=lease.requested_at,
                job_id=lease.job_id,
                run_token=lease.run_token,
                island_id=None,
                commit_hash=None,
                ordinal=None,
                duration_seconds=None,
                payload=payload,
            )
        )
        if lease.acquired_at is not None:
            records.append(
                _event_record(
                    event_id=f"evaluation_lease:{lease.id}:acquired",
                    source="evaluation_resource_lease",
                    event_type="evaluation.lease.acquired",
                    occurred_at=lease.acquired_at,
                    job_id=lease.job_id,
                    run_token=lease.run_token,
                    island_id=None,
                    commit_hash=None,
                    ordinal=None,
                    duration_seconds=lease.wait_seconds,
                    payload=payload,
                )
            )
        if lease.released_at is not None:
            records.append(
                _event_record(
                    event_id=f"evaluation_lease:{lease.id}:released",
                    source="evaluation_resource_lease",
                    event_type="evaluation.lease.released",
                    occurred_at=lease.released_at,
                    job_id=lease.job_id,
                    run_token=lease.run_token,
                    island_id=None,
                    commit_hash=None,
                    ordinal=None,
                    duration_seconds=_elapsed_seconds(
                        lease.acquired_at,
                        lease.released_at,
                    ),
                    payload={
                        **payload,
                        "release_reason": lease.release_reason,
                    },
                )
            )
    return records


def _derive_interrupted_stages(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    starts: dict[tuple[str, str | None, str, int], Mapping[str, object]] = {}
    closed: set[tuple[str, str | None, str, int]] = set()
    interruptions: dict[tuple[str, str | None], Mapping[str, object]] = {}
    start_type_to_stage = {pair[0]: stage for stage, pair in _STAGE_TYPES.items()}
    finish_type_to_stage = {pair[1]: stage for stage, pair in _STAGE_TYPES.items()}
    for record in records:
        event_type = str(record.get("event_type") or "")
        job_id = str(record.get("job_id") or "")
        run_token = _optional_text(record.get("run_token"))
        ordinal = int(record.get("ordinal") or 0)
        stage = start_type_to_stage.get(event_type)
        if stage is not None and job_id and ordinal > 0:
            starts[(job_id, run_token, stage, ordinal)] = record
        stage = finish_type_to_stage.get(event_type)
        if stage is not None and job_id and ordinal > 0:
            closed.add((job_id, run_token, stage, ordinal))
        if event_type in _RUN_INTERRUPTION_EVENT_TYPES and job_id:
            interruptions[(job_id, run_token)] = record

    ingestion_retries: dict[tuple[str, str | None, str, int], Mapping[str, object]] = {}
    ingestion_starts_by_job: dict[
        str,
        list[
            tuple[
                tuple[str, str | None, str, int],
                Mapping[str, object],
            ]
        ],
    ] = {}
    for key, start in starts.items():
        if key[2] == "ingestion":
            ingestion_starts_by_job.setdefault(key[0], []).append((key, start))
    for attempts in ingestion_starts_by_job.values():
        ordered_attempts = sorted(
            attempts,
            key=lambda item: (item[0][3], *_timeline_sort_key(item[1])),
        )
        for key, _start in ordered_attempts:
            retry = next(
                (
                    candidate
                    for candidate in ordered_attempts
                    if candidate[0][3] > key[3]
                ),
                None,
            )
            if retry is not None:
                ingestion_retries[key] = retry[1]

    derived: list[dict[str, object]] = []
    for key, start in starts.items():
        if key in closed:
            continue
        interruption = interruptions.get((key[0], key[1]))
        payload: dict[str, object]
        if key[2] == "ingestion":
            interruption = ingestion_retries.get(key)
            if interruption is None:
                continue
            payload = {
                "reason": "superseded_by_retry",
                "superseding_event_id": str(interruption.get("event_id") or ""),
            }
        else:
            if interruption is None:
                continue
            payload = {
                "reason": str(interruption.get("event_type") or "run_terminated"),
                "terminal_event_id": str(interruption.get("event_id") or ""),
            }
        interrupted_at = _parse_timestamp(interruption.get("occurred_at"))
        start_at = _parse_timestamp(start.get("occurred_at"))
        derived.append(
            _event_record(
                event_id=f"derived:{key[0]}:{key[1]}:{key[2]}:{key[3]}:interrupted",
                source="derived",
                event_type=f"{key[2]}.interrupted",
                occurred_at=interrupted_at,
                job_id=key[0],
                run_token=key[1],
                island_id=_optional_text(start.get("island_id")),
                commit_hash=_optional_text(start.get("commit_hash")),
                ordinal=key[3],
                duration_seconds=_elapsed_seconds(start_at, interrupted_at),
                payload=payload,
            )
        )
    return derived


def _timeline_issues(
    records: Sequence[Mapping[str, object]],
    *,
    jobs: Sequence[EvolutionJob],
) -> Iterable[TimelineIssue]:
    yield from _duration_and_order_issues(records)
    yield from _job_terminal_issues(records, jobs=jobs)
    yield from _stage_pairing_issues(records, jobs=jobs)
    yield from _archive_membership_issues(records)


def _duration_and_order_issues(
    records: Sequence[Mapping[str, object]],
) -> Iterable[TimelineIssue]:
    for record in records:
        duration = record.get("duration_seconds")
        if duration is None:
            continue
        try:
            parsed = float(duration)
        except (TypeError, ValueError):
            parsed = math.nan
        if not math.isfinite(parsed) or parsed < 0:
            yield TimelineIssue(
                code="invalid_duration",
                message=(f"event {record.get('event_id')} has an impossible duration"),
                job_id=_optional_text(record.get("job_id")),
                event_id=_optional_text(record.get("event_id")),
            )


def _job_terminal_issues(
    records: Sequence[Mapping[str, object]],
    *,
    jobs: Sequence[EvolutionJob],
) -> Iterable[TimelineIssue]:
    boundary = min(
        (
            _parse_timestamp(record.get("occurred_at"))
            for record in records
            if record.get("event_type") == TIMELINE_HISTORY_BOUNDARY
        ),
        default=None,
    )
    eligible_job_ids = {
        str(job.id)
        for job in jobs
        if boundary is None or _aware_utc(job.created_at) >= boundary
    }
    run_started_keys = {
        (
            str(record.get("job_id")),
            _optional_text(record.get("run_token")),
        )
        for record in records
        if record.get("event_type") == JOB_RUN_STARTED and record.get("job_id")
    }
    run_started_jobs = {key[0] for key in run_started_keys}
    terminal_event_jobs = {
        str(record.get("job_id"))
        for record in records
        if record.get("event_type") in _TERMINAL_EVENT_TYPES and record.get("job_id")
    }
    pre_run_cancellations = {
        str(record.get("job_id"))
        for record in records
        if record.get("event_type") == JOB_CANCELLED
        and str(dict(record.get("payload") or {}).get("previous_status") or "")
        in {"staged", "pending"}
    }
    checked_terminal_runs: set[tuple[str, str | None]] = set()
    for record in records:
        if record.get("event_type") not in _TERMINAL_EVENT_TYPES:
            continue
        job_id = _optional_text(record.get("job_id"))
        if not job_id or job_id not in eligible_job_ids:
            continue
        payload = dict(record.get("payload") or {})
        if record.get("event_type") == JOB_CANCELLED and str(
            payload.get("previous_status") or ""
        ) in {"staged", "pending"}:
            continue
        run_key = (job_id, _optional_text(record.get("run_token")))
        if run_key in checked_terminal_runs:
            continue
        checked_terminal_runs.add(run_key)
        if run_key not in run_started_keys:
            run_token = run_key[1] or "<none>"
            yield TimelineIssue(
                code="terminal_without_run_start",
                message=(
                    f"terminal run {job_id}/{run_token} has no matching "
                    "job.run.started event"
                ),
                job_id=job_id,
                event_id=_optional_text(record.get("event_id")),
            )

    for job in jobs:
        status = str(getattr(job.status, "value", job.status) or "").lower()
        if status not in {"succeeded", "failed", "cancelled"}:
            continue
        created_at = _aware_utc(job.created_at)
        if boundary is not None and created_at < boundary:
            continue
        job_id = str(job.id)
        if job_id not in terminal_event_jobs:
            if job_id not in run_started_jobs and job_id not in pre_run_cancellations:
                yield TimelineIssue(
                    code="terminal_without_run_start",
                    message=(f"terminal job {job_id} has no recorded worker run start"),
                    job_id=job_id,
                )
            yield TimelineIssue(
                code="terminal_without_terminal_event",
                message=f"terminal job {job_id} has no append-only terminal event",
                job_id=job_id,
            )


def _stage_pairing_issues(
    records: Sequence[Mapping[str, object]],
    *,
    jobs: Sequence[EvolutionJob],
) -> Iterable[TimelineIssue]:
    start_by_key: dict[tuple[str, str | None, str, int], Mapping[str, object]] = {}
    finish_by_key: dict[tuple[str, str | None, str, int], Mapping[str, object]] = {}
    interrupted_keys: set[tuple[str, str | None, str, int]] = set()
    job_by_id = {str(job.id): job for job in jobs}
    start_types = {pair[0]: stage for stage, pair in _STAGE_TYPES.items()}
    finish_types = {pair[1]: stage for stage, pair in _STAGE_TYPES.items()}
    for record in records:
        event_type = str(record.get("event_type") or "")
        job_id = _optional_text(record.get("job_id"))
        ordinal = int(record.get("ordinal") or 0)
        if not job_id or ordinal <= 0:
            continue
        run_token = _optional_text(record.get("run_token"))
        stage = start_types.get(event_type)
        if stage is not None:
            start_by_key[(job_id, run_token, stage, ordinal)] = record
            continue
        stage = finish_types.get(event_type)
        if stage is not None:
            finish_by_key[(job_id, run_token, stage, ordinal)] = record
            continue
        if event_type.endswith(".interrupted"):
            stage = event_type.removesuffix(".interrupted")
            interrupted_keys.add((job_id, run_token, stage, ordinal))

    for key, finish in finish_by_key.items():
        start = start_by_key.get(key)
        if start is None:
            yield TimelineIssue(
                code="stage_finish_without_start",
                message=(
                    f"{key[2]} finish for job {key[0]} ordinal {key[3]} "
                    "has no matching start"
                ),
                job_id=key[0],
                event_id=_optional_text(finish.get("event_id")),
            )
            continue
        if _parse_timestamp(finish.get("occurred_at")) < _parse_timestamp(
            start.get("occurred_at")
        ):
            yield TimelineIssue(
                code="stage_finish_before_start",
                message=(
                    f"{key[2]} finish precedes start for job {key[0]} ordinal {key[3]}"
                ),
                job_id=key[0],
                event_id=_optional_text(finish.get("event_id")),
            )
    for key, start in start_by_key.items():
        if key in finish_by_key or key in interrupted_keys:
            continue
        job = job_by_id.get(key[0])
        if job is None:
            continue
        if key[2] == "ingestion":
            ingestion_status = str(getattr(job, "ingestion_status", "") or "").lower()
            if ingestion_status not in {"succeeded", "skipped"}:
                continue
        else:
            status = str(getattr(job.status, "value", job.status) or "").lower()
            if status not in {"succeeded", "failed", "cancelled"}:
                continue
        yield TimelineIssue(
            code="active_stage_without_interruption",
            message=(
                f"{key[2]} start for job {key[0]} ordinal {key[3]} "
                "is not finished or explicitly interrupted"
            ),
            job_id=key[0],
            event_id=_optional_text(start.get("event_id")),
        )


def _archive_membership_issues(
    records: Sequence[Mapping[str, object]],
) -> Iterable[TimelineIssue]:
    membership: dict[tuple[str, str], int] = {}
    for record in records:
        event_type = str(record.get("event_type") or "")
        if event_type not in {
            ARCHIVE_MEMBER_INITIAL_STATE,
            ARCHIVE_MEMBER_ADMITTED,
            ARCHIVE_MEMBER_MOVED,
            ARCHIVE_MEMBER_REMOVED,
        }:
            continue
        island = _optional_text(record.get("island_id"))
        commit = _optional_text(record.get("commit_hash"))
        if not island or not commit:
            continue
        key = (island, commit)
        payload = dict(record.get("payload") or {})
        if event_type == ARCHIVE_MEMBER_INITIAL_STATE:
            membership[key] = int(payload.get("cell_index") or 0)
        elif event_type == ARCHIVE_MEMBER_ADMITTED:
            membership[key] = int(payload.get("to_cell") or 0)
        elif event_type == ARCHIVE_MEMBER_MOVED:
            if key not in membership:
                yield TimelineIssue(
                    code="archive_move_without_membership",
                    message=(
                        f"archive move for {island}/{commit} has no prior membership"
                    ),
                    job_id=_optional_text(record.get("job_id")),
                    event_id=_optional_text(record.get("event_id")),
                )
            membership[key] = int(payload.get("to_cell") or 0)
        elif event_type == ARCHIVE_MEMBER_REMOVED:
            if key not in membership:
                yield TimelineIssue(
                    code="archive_removal_without_membership",
                    message=(
                        f"archive removal for {island}/{commit} has no prior membership"
                    ),
                    job_id=_optional_text(record.get("job_id")),
                    event_id=_optional_text(record.get("event_id")),
                )
            membership.pop(key, None)


def _event_record(
    *,
    event_id: str,
    source: str,
    event_type: str,
    occurred_at: datetime,
    job_id: UUID | str | None,
    run_token: UUID | str | None,
    island_id: str | None,
    commit_hash: str | None,
    ordinal: int | None,
    duration_seconds: float | None,
    payload: Mapping[str, object],
) -> dict[str, object]:
    return {
        "record_type": "event",
        "timeline_schema_version": TIMELINE_SCHEMA_VERSION,
        "event_id": str(event_id),
        "source": str(source),
        "event_type": str(event_type),
        "occurred_at": _iso_utc(occurred_at),
        "job_id": str(job_id) if job_id is not None else None,
        "run_token": str(run_token) if run_token is not None else None,
        "island_id": str(island_id) if island_id is not None else None,
        "commit_hash": str(commit_hash) if commit_hash is not None else None,
        "ordinal": int(ordinal) if ordinal is not None else None,
        "duration_seconds": _json_duration(duration_seconds),
        "payload": {
            str(key): _json_safe_value(value)
            for key, value in sorted(payload.items())
            if value is not None
        },
    }


def _timeline_sort_key(record: Mapping[str, object]) -> tuple[str, str]:
    return (
        str(record.get("occurred_at") or ""),
        str(record.get("event_id") or ""),
    )


def _json_safe_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return _iso_utc(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _json_duration(value: float | None) -> float | str | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else str(parsed)


def _elapsed_seconds(
    started_at: datetime | None,
    finished_at: datetime | None,
) -> float | None:
    if started_at is None or finished_at is None:
        return None
    return (_aware_utc(finished_at) - _aware_utc(started_at)).total_seconds()


def _parse_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return _aware_utc(value)
    raw = str(value or "").strip()
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    parsed = datetime.fromisoformat(raw)
    return _aware_utc(parsed)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _iso_utc(value: datetime) -> str:
    return _aware_utc(value).isoformat().replace("+00:00", "Z")


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


__all__ = [
    "TIMELINE_SCHEMA_VERSION",
    "EvolutionTimelineExport",
    "TimelineCompletenessError",
    "TimelineIssue",
    "export_evolution_timeline",
]

"""Bounded, append-only evolution event recording primitives."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from loreley.db.models import EvolutionEvent

TIMELINE_HISTORY_BOUNDARY = "timeline.history_boundary"

JOB_DISPATCHED = "job.dispatched"
JOB_RUN_STARTED = "job.run.started"
JOB_REQUEUED = "job.requeued"
JOB_RECLAIMED = "job.reclaimed"
JOB_RECOVERY_EXHAUSTED = "job.recovery_exhausted"
JOB_SUCCEEDED = "job.succeeded"
JOB_FAILED = "job.failed"
JOB_CANCELLED = "job.cancelled"

PLANNING_STARTED = "planning.started"
PLANNING_FINISHED = "planning.finished"
CODING_STARTED = "coding.started"
CODING_FINISHED = "coding.finished"
EVALUATION_INVOCATION_STARTED = "evaluation.invocation.started"
INGESTION_STARTED = "ingestion.started"
INGESTION_FINISHED = "ingestion.finished"

ARCHIVE_CANDIDATE_CONSIDERED = "archive.candidate.considered"
ARCHIVE_MEMBER_INITIAL_STATE = "archive.member.initial_state"
ARCHIVE_MEMBER_ADMITTED = "archive.member.admitted"
ARCHIVE_MEMBER_MOVED = "archive.member.moved"
ARCHIVE_MEMBER_REMOVED = "archive.member.removed"
ARCHIVE_REBUILD_COMPLETED = "archive.rebuild.completed"

_EVENT_TYPE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_PAYLOAD_MAX_BYTES = 4096
_PAYLOAD_STRING_MAX_CHARS = 256
_PAYLOAD_SEQUENCE_MAX_ITEMS = 32

_EVENT_PAYLOAD_FIELDS: dict[str, frozenset[str]] = {
    TIMELINE_HISTORY_BOUNDARY: frozenset({"reason", "schema_from", "schema_to"}),
    JOB_DISPATCHED: frozenset({"dispatch_kind", "previous_status", "recovery_count"}),
    JOB_RUN_STARTED: frozenset({"job_kind", "recovery_count"}),
    JOB_REQUEUED: frozenset({"reason", "previous_status", "recovery_count", "manual"}),
    JOB_RECLAIMED: frozenset({"reason", "outcome", "recovery_count"}),
    JOB_RECOVERY_EXHAUSTED: frozenset({"failure_kind", "recovery_count"}),
    JOB_SUCCEEDED: frozenset({"outcome"}),
    JOB_FAILED: frozenset({"failure_stage", "failure_kind"}),
    JOB_CANCELLED: frozenset({"reason", "previous_status"}),
    PLANNING_STARTED: frozenset(),
    PLANNING_FINISHED: frozenset({"outcome", "failure_kind"}),
    CODING_STARTED: frozenset({"rework"}),
    CODING_FINISHED: frozenset({"outcome", "failure_kind", "rework"}),
    EVALUATION_INVOCATION_STARTED: frozenset(
        {"evaluator_name", "evaluator_version", "protocol"}
    ),
    INGESTION_STARTED: frozenset(),
    INGESTION_FINISHED: frozenset({"outcome", "reason", "status_code", "inserted"}),
    ARCHIVE_CANDIDATE_CONSIDERED: frozenset({"outcome", "reason", "cell_index"}),
    ARCHIVE_MEMBER_INITIAL_STATE: frozenset({"cell_index", "reason"}),
    ARCHIVE_MEMBER_ADMITTED: frozenset({"to_cell", "reason"}),
    ARCHIVE_MEMBER_MOVED: frozenset({"from_cell", "to_cell", "reason"}),
    ARCHIVE_MEMBER_REMOVED: frozenset({"from_cell", "reason"}),
    ARCHIVE_REBUILD_COMPLETED: frozenset(
        {
            "reason",
            "before_count",
            "after_count",
            "admitted_count",
            "moved_count",
            "removed_count",
            "projection_epoch",
        }
    ),
}

_STAGE_EVENT_TYPES = {
    "planning": (PLANNING_STARTED, PLANNING_FINISHED),
    "coding": (CODING_STARTED, CODING_FINISHED),
    "evaluation": (EVALUATION_INVOCATION_STARTED, None),
    "ingestion": (INGESTION_STARTED, INGESTION_FINISHED),
}


class EvolutionEventValidationError(ValueError):
    """Raised before an unsafe or malformed event can reach the database."""


@dataclass(frozen=True, slots=True)
class EvolutionEventReceipt:
    event_id: UUID
    event_key: str
    event_type: str
    occurred_at: datetime
    ordinal: int | None
    inserted: bool


@dataclass(frozen=True, slots=True)
class EvolutionStageHandle:
    stage: str
    job_id: UUID
    run_token: UUID | None
    island_id: str | None
    commit_hash: str | None
    ordinal: int
    started_at: datetime
    monotonic_started: float

    @property
    def event_key_prefix(self) -> str:
        run = str(self.run_token) if self.run_token is not None else "none"
        return f"job:{self.job_id}:run:{run}:stage:{self.stage}:{self.ordinal}"


def evolution_event_key(
    event_type: str,
    *,
    job_id: UUID | str | None = None,
    run_token: UUID | str | None = None,
    ordinal: int | None = None,
    key_parts: Sequence[object] = (),
) -> str:
    """Return a deterministic, opaque key for one logical event delivery."""

    canonical = json.dumps(
        {
            "version": 1,
            "event_type": str(event_type),
            "job_id": str(job_id) if job_id is not None else None,
            "run_token": str(run_token) if run_token is not None else None,
            "ordinal": ordinal,
            "key_parts": [str(part) for part in key_parts],
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"evo1:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def next_event_ordinal(
    session: Session | Any,
    *,
    event_type: str,
    job_id: UUID | None = None,
    run_token: UUID | None = None,
    island_id: str | None = None,
) -> int:
    """Return the next positive ordinal in a serialized event scope."""

    _validate_event_type(event_type)
    if not _is_sqlalchemy_session(session):
        return 1
    stmt = select(func.coalesce(func.max(EvolutionEvent.ordinal), 0) + 1).where(
        EvolutionEvent.event_type == event_type
    )
    if job_id is not None:
        stmt = stmt.where(EvolutionEvent.job_id == job_id)
    if run_token is not None:
        stmt = stmt.where(EvolutionEvent.run_token == run_token)
    if island_id is not None:
        stmt = stmt.where(EvolutionEvent.island_id == island_id)
    return int(session.execute(stmt).scalar_one())


def record_evolution_event(
    session: Session | Any,
    *,
    event_type: str,
    job_id: UUID | str | None = None,
    run_token: UUID | str | None = None,
    island_id: str | None = None,
    commit_hash: str | None = None,
    occurred_at: datetime | None = None,
    ordinal: int | None = None,
    duration_seconds: float | None = None,
    payload: Mapping[str, object] | None = None,
    event_key: str | None = None,
    key_parts: Sequence[object] = (),
) -> EvolutionEventReceipt:
    """Insert one event with PostgreSQL conflict handling.

    Database failures intentionally propagate to the state-transition caller.
    """

    normalized_type = _validate_event_type(event_type)
    normalized_job_id = _uuid_or_none(job_id, field="job_id")
    normalized_run_token = _uuid_or_none(run_token, field="run_token")
    normalized_island = _bounded_identifier(island_id, field="island_id", limit=64)
    normalized_commit = _bounded_identifier(
        commit_hash,
        field="commit_hash",
        limit=64,
    )
    normalized_at = _aware_utc(occurred_at or datetime.now(UTC))
    normalized_ordinal = _positive_ordinal_or_none(ordinal)
    normalized_duration = _duration_or_none(duration_seconds)
    normalized_payload = sanitized_event_payload(normalized_type, payload or {})
    normalized_key = str(event_key or "").strip() or evolution_event_key(
        normalized_type,
        job_id=normalized_job_id,
        run_token=normalized_run_token,
        ordinal=normalized_ordinal,
        key_parts=key_parts,
    )
    if len(normalized_key) > 255:
        raise EvolutionEventValidationError("event_key exceeds 255 characters")

    event_id = uuid4()
    if not _is_sqlalchemy_session(session):
        return _record_with_test_double(
            session,
            event_id=event_id,
            event_key=normalized_key,
            event_type=normalized_type,
            job_id=normalized_job_id,
            run_token=normalized_run_token,
            island_id=normalized_island,
            commit_hash=normalized_commit,
            occurred_at=normalized_at,
            ordinal=normalized_ordinal,
            duration_seconds=normalized_duration,
            payload=normalized_payload,
        )

    statement = (
        pg_insert(EvolutionEvent)
        .values(
            id=event_id,
            event_key=normalized_key,
            event_type=normalized_type,
            job_id=normalized_job_id,
            run_token=normalized_run_token,
            island_id=normalized_island,
            commit_hash=normalized_commit,
            occurred_at=normalized_at,
            ordinal=normalized_ordinal,
            duration_seconds=normalized_duration,
            payload=normalized_payload,
        )
        .on_conflict_do_nothing(index_elements=[EvolutionEvent.event_key])
        .returning(EvolutionEvent.id)
    )
    inserted_id = session.execute(statement).scalar_one_or_none()
    if inserted_id is not None:
        return EvolutionEventReceipt(
            event_id=inserted_id,
            event_key=normalized_key,
            event_type=normalized_type,
            occurred_at=normalized_at,
            ordinal=normalized_ordinal,
            inserted=True,
        )

    existing = session.execute(
        select(EvolutionEvent).where(EvolutionEvent.event_key == normalized_key)
    ).scalar_one()
    return EvolutionEventReceipt(
        event_id=existing.id,
        event_key=existing.event_key,
        event_type=existing.event_type,
        occurred_at=_aware_utc(existing.occurred_at),
        ordinal=existing.ordinal,
        inserted=False,
    )


def start_evolution_stage(
    session: Session | Any,
    *,
    stage: str,
    job_id: UUID,
    run_token: UUID | None = None,
    island_id: str | None = None,
    commit_hash: str | None = None,
    ordinal: int | None = None,
    occurred_at: datetime | None = None,
    payload: Mapping[str, object] | None = None,
) -> EvolutionStageHandle:
    """Persist a stage start before the external or long-running operation."""

    normalized_stage = str(stage or "").strip().lower()
    pair = _STAGE_EVENT_TYPES.get(normalized_stage)
    if pair is None:
        raise EvolutionEventValidationError(f"unsupported evolution stage {stage!r}")
    start_type = pair[0]
    effective_ordinal = ordinal or next_event_ordinal(
        session,
        event_type=start_type,
        job_id=job_id,
        run_token=run_token,
    )
    started_at = _aware_utc(occurred_at or datetime.now(UTC))
    started_monotonic = monotonic()
    receipt = record_evolution_event(
        session,
        event_type=start_type,
        job_id=job_id,
        run_token=run_token,
        island_id=island_id,
        commit_hash=commit_hash,
        occurred_at=started_at,
        ordinal=effective_ordinal,
        payload=payload,
        key_parts=("stage", normalized_stage),
    )
    return EvolutionStageHandle(
        stage=normalized_stage,
        job_id=job_id,
        run_token=run_token,
        island_id=island_id,
        commit_hash=commit_hash,
        ordinal=int(receipt.ordinal or effective_ordinal),
        started_at=receipt.occurred_at,
        monotonic_started=started_monotonic,
    )


def finish_evolution_stage(
    session: Session | Any,
    *,
    handle: EvolutionStageHandle,
    outcome: str,
    failure_kind: str | None = None,
    payload: Mapping[str, object] | None = None,
    occurred_at: datetime | None = None,
) -> EvolutionEventReceipt:
    """Close a planning, coding, or ingestion stage with monotonic duration."""

    finish_type = _STAGE_EVENT_TYPES[handle.stage][1]
    if finish_type is None:
        raise EvolutionEventValidationError(
            f"stage {handle.stage!r} closes through another authoritative record"
        )
    merged_payload = dict(payload or {})
    merged_payload["outcome"] = _bounded_text(outcome)
    if failure_kind:
        merged_payload["failure_kind"] = _bounded_text(failure_kind)
    return record_evolution_event(
        session,
        event_type=finish_type,
        job_id=handle.job_id,
        run_token=handle.run_token,
        island_id=handle.island_id,
        commit_hash=handle.commit_hash,
        occurred_at=occurred_at,
        ordinal=handle.ordinal,
        duration_seconds=max(0.0, monotonic() - handle.monotonic_started),
        payload=merged_payload,
        key_parts=("stage", handle.stage),
    )


def sanitized_event_payload(
    event_type: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Validate an explicit event-specific allowlist and bound JSON size."""

    allowed = _EVENT_PAYLOAD_FIELDS.get(event_type)
    if allowed is None:
        raise EvolutionEventValidationError(
            f"event_type {event_type!r} has no payload allowlist"
        )
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise EvolutionEventValidationError(
            f"event_type {event_type!r} rejected payload fields: {unknown}"
        )
    sanitized = {
        str(key): _sanitized_payload_value(value)
        for key, value in payload.items()
        if value is not None
    }
    encoded = json.dumps(
        sanitized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > _PAYLOAD_MAX_BYTES:
        raise EvolutionEventValidationError(
            f"event payload exceeds {_PAYLOAD_MAX_BYTES} bytes"
        )
    return sanitized


def bounded_failure_kind(error: BaseException, *, fallback: str) -> str:
    """Return a stable class-level failure label without exception text."""

    raw = error.__class__.__name__ or fallback
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", raw).lower()
    normalized = re.sub(r"[^a-z0-9_.-]+", "_", snake).strip("_")
    return (normalized or fallback)[:64]


def _record_with_test_double(
    session: Any,
    **values: Any,
) -> EvolutionEventReceipt:
    """Keep lightweight unit-test sessions usable without weakening DB writes."""

    add = getattr(session, "add", None)
    if callable(add):
        add(
            EvolutionEvent(
                id=values["event_id"],
                event_key=values["event_key"],
                event_type=values["event_type"],
                job_id=values["job_id"],
                run_token=values["run_token"],
                island_id=values["island_id"],
                commit_hash=values["commit_hash"],
                occurred_at=values["occurred_at"],
                ordinal=values["ordinal"],
                duration_seconds=values["duration_seconds"],
                payload=values["payload"],
            )
        )
    return EvolutionEventReceipt(
        event_id=values["event_id"],
        event_key=values["event_key"],
        event_type=values["event_type"],
        occurred_at=values["occurred_at"],
        ordinal=values["ordinal"],
        inserted=True,
    )


def _is_sqlalchemy_session(session: Any) -> bool:
    return isinstance(session, Session)


def _validate_event_type(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _EVENT_TYPE_PATTERN.fullmatch(normalized):
        raise EvolutionEventValidationError(f"invalid event_type {value!r}")
    if normalized not in _EVENT_PAYLOAD_FIELDS:
        raise EvolutionEventValidationError(f"unsupported event_type {value!r}")
    return normalized


def _uuid_or_none(value: UUID | str | None, *, field: str) -> UUID | None:
    if value is None or isinstance(value, UUID):
        return value
    try:
        return UUID(str(value).strip())
    except (ValueError, AttributeError) as exc:
        raise EvolutionEventValidationError(f"invalid {field}") from exc


def _bounded_identifier(value: str | None, *, field: str, limit: int) -> str | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    if len(normalized) > limit:
        raise EvolutionEventValidationError(f"{field} exceeds {limit} characters")
    return normalized


def _aware_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EvolutionEventValidationError("occurred_at must be timezone-aware")
    return value.astimezone(UTC)


def _positive_ordinal_or_none(value: int | None) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise EvolutionEventValidationError("ordinal must be positive")
    return parsed


def _duration_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise EvolutionEventValidationError(
            "duration_seconds must be finite and non-negative"
        )
    return parsed


def _bounded_text(value: object) -> str:
    normalized = " ".join(str(value or "").split())
    return normalized[:_PAYLOAD_STRING_MAX_CHARS]


def _sanitized_payload_value(value: object) -> object:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvolutionEventValidationError("payload floats must be finite")
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, str):
        return _bounded_text(value)
    if isinstance(value, (list, tuple)):
        if len(value) > _PAYLOAD_SEQUENCE_MAX_ITEMS:
            raise EvolutionEventValidationError(
                f"payload sequences are limited to {_PAYLOAD_SEQUENCE_MAX_ITEMS} items"
            )
        return [_sanitized_payload_value(item) for item in value]
    raise EvolutionEventValidationError(
        f"unsupported event payload value type {type(value).__name__}"
    )


__all__ = [
    "ARCHIVE_CANDIDATE_CONSIDERED",
    "ARCHIVE_MEMBER_ADMITTED",
    "ARCHIVE_MEMBER_INITIAL_STATE",
    "ARCHIVE_MEMBER_MOVED",
    "ARCHIVE_MEMBER_REMOVED",
    "ARCHIVE_REBUILD_COMPLETED",
    "CODING_FINISHED",
    "CODING_STARTED",
    "EVALUATION_INVOCATION_STARTED",
    "INGESTION_FINISHED",
    "INGESTION_STARTED",
    "JOB_CANCELLED",
    "JOB_DISPATCHED",
    "JOB_FAILED",
    "JOB_RECLAIMED",
    "JOB_RECOVERY_EXHAUSTED",
    "JOB_REQUEUED",
    "JOB_RUN_STARTED",
    "JOB_SUCCEEDED",
    "PLANNING_FINISHED",
    "PLANNING_STARTED",
    "TIMELINE_HISTORY_BOUNDARY",
    "EvolutionEventReceipt",
    "EvolutionEventValidationError",
    "EvolutionStageHandle",
    "bounded_failure_kind",
    "evolution_event_key",
    "finish_evolution_stage",
    "next_event_ordinal",
    "record_evolution_event",
    "sanitized_event_payload",
    "start_evolution_stage",
]

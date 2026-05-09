"""Agent REST facade orchestration and audit persistence."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi.encoders import jsonable_encoder
from loguru import logger
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from loreley.api.agent_errors import AgentAPIError
from loreley.api.schemas.agent import AgentActionRequest
from loreley.api.services.evidence import (
    build_agent_feedback_payload,
    build_evaluation_artifact_payload,
)
from loreley.api.services.jobs import (
    JobNotFoundError,
    JobRetryConflictError,
    JobRetryValidationError,
    retry_failed_stale_jobs,
    retry_job_by_id,
)
from loreley.api.services.operator import (
    OperatorTaskAlreadyActiveError,
    create_baseline_ensure_task,
    operator_status,
    run_baseline_ensure_task,
)
from loreley.api.services.repair import (
    RepairConflictError,
    RepairNotFoundError,
    RepairValidationError,
    schedule_one_repair,
    update_candidate_operator_state,
)
from loreley.config import Settings, get_settings
from loreley.core.contracts import clamp_text, normalize_single_line
from loreley.core.job_retry import (
    job_lease_payload,
    job_retry_state,
    job_status_value,
    load_failed_stale_retry_rows,
)
from loreley.db.base import INSTANCE_SCHEMA_VERSION, session_scope
from loreley.db.models import AgentAction, CandidateCommit, EvolutionJob, JobStatus

log = logger.bind(module="api.agent")

AGENT_SCHEMA_VERSION = "agent-rest-control-facade.v1"
_MAX_ACTION_TYPE_CHARS = 64
_ACTION_STATUS_SUCCEEDED = "succeeded"
_ACTION_STATUS_FAILED = "failed"
_ACTIVE_JOB_STATUSES = (JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING)
_ACTION_RISK: dict[str, str] = {
    "retry_job": "medium",
    "retry_failed_stale_jobs": "high",
    "baseline_ensure": "medium",
    "repair_schedule_one": "medium",
    "repair_candidate_quarantine": "medium",
    "repair_candidate_discard": "high",
    "repair_candidate_restore": "medium",
}
_CANDIDATE_ACTIONS: dict[str, str] = {
    "repair_candidate_quarantine": "quarantine",
    "repair_candidate_discard": "discard",
    "repair_candidate_restore": "restore",
}


def agent_capabilities(*, settings: Settings | None = None) -> dict[str, Any]:
    active_settings = settings or get_settings()
    token_configured = bool(str(active_settings.loreley_agent_api_token or "").strip())
    return {
        "schema_version": AGENT_SCHEMA_VERSION,
        "database_schema_version": INSTANCE_SCHEMA_VERSION,
        "auth": {
            "type": "bearer",
            "configured": token_configured,
            "optional_when_unset": True,
            "environment_variable": "LORELEY_AGENT_API_TOKEN",
        },
        "read_resources": [
            {"resource": "operator_status", "path": "/api/v1/agent/status"},
            {"resource": "next_actions", "path": "/api/v1/agent/next-actions"},
            {"resource": "job_feedback", "path": "/api/v1/agent/jobs/{job_id}/feedback"},
            {
                "resource": "commit_feedback",
                "path": "/api/v1/agent/commits/{commit_hash}/feedback",
            },
            {"resource": "action_audit", "path": "/api/v1/agent/actions/{action_id}"},
        ],
        "actions": [
            _capability_action(
                "retry_job",
                required_params=["job_id"],
                expected_state_fields=["status", "lease_state", "recovery_count"],
            ),
            _capability_action(
                "retry_failed_stale_jobs",
                required_params=["all or limit"],
                expected_state_fields=[],
            ),
            _capability_action(
                "baseline_ensure",
                expected_state_fields=["campaign_program_hash", "baseline_status"],
            ),
            _capability_action("repair_schedule_one"),
            _capability_action(
                "repair_candidate_quarantine",
                required_params=["candidate_id"],
                expected_state_fields=[
                    "lifecycle_status",
                    "repair_state",
                    "active_repair_job_id",
                ],
            ),
            _capability_action(
                "repair_candidate_discard",
                required_params=["candidate_id"],
                expected_state_fields=[
                    "lifecycle_status",
                    "repair_state",
                    "active_repair_job_id",
                ],
            ),
            _capability_action(
                "repair_candidate_restore",
                required_params=["candidate_id"],
                expected_state_fields=[
                    "lifecycle_status",
                    "repair_state",
                    "active_repair_job_id",
                ],
            ),
        ],
        "error_shape": {
            "error_code": "string",
            "message": "string",
            "retryable": "boolean",
            "resource": {"type": "string", "id": "string"},
            "suggested_next_actions": [],
        },
    }


def agent_status(*, settings: Settings | None = None) -> dict[str, Any]:
    status_payload = operator_status(settings=settings) if settings is not None else operator_status()
    blocking_issues = blocking_issues_from_operator_status(status_payload)
    next_actions = next_actions_from_operator_status(
        status_payload,
        settings=settings,
        blocking_issues=blocking_issues,
    )
    if blocking_issues:
        health = "blocked"
    elif next_actions:
        health = "actionable"
    else:
        health = "healthy"
    return {
        "operator_status": status_payload,
        "health": health,
        "blocking_issues": blocking_issues,
        "safe_next_actions": next_actions,
    }


def agent_next_actions(*, settings: Settings | None = None) -> list[dict[str, Any]]:
    status_payload = operator_status(settings=settings) if settings is not None else operator_status()
    blocking_issues = blocking_issues_from_operator_status(status_payload)
    return next_actions_from_operator_status(
        status_payload,
        settings=settings,
        blocking_issues=blocking_issues,
    )


def blocking_issues_from_operator_status(status_payload: dict[str, Any]) -> list[dict[str, Any]]:
    scheduler = _nested_dict(status_payload, "campaign_program", "scheduler")
    current_matches_active = scheduler.get("current_matches_active")
    if current_matches_active is not False:
        return []
    return [
        {
            "issue_type": "campaign_program_mismatch",
            "message": "Active campaign program does not match the current repository file.",
            "resource": {
                "type": "campaign_program",
                "id": str(scheduler.get("active_hash") or "active"),
            },
            "suggested_next_actions": [],
        }
    ]


def next_actions_from_operator_status(
    status_payload: dict[str, Any],
    *,
    settings: Settings | None = None,
    blocking_issues: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if blocking_issues is None:
        blocking_issues = blocking_issues_from_operator_status(status_payload)
    if blocking_issues:
        return []

    active_settings = settings or get_settings()
    actions: list[dict[str, Any]] = []
    for action in (
        _failed_stale_next_action(status_payload),
        _baseline_next_action(status_payload),
        _repair_next_action(status_payload, settings=active_settings),
    ):
        if action is not None:
            actions.append(action)
    return actions


def _failed_stale_next_action(status_payload: dict[str, Any]) -> dict[str, Any] | None:
    job_leases = _nested_dict(status_payload, "job_health", "job_leases")
    failed_stale_count = _safe_int(job_leases.get("recovery_exhausted_failed"))
    if failed_stale_count <= 0:
        return None
    limit = min(failed_stale_count, 50)
    return {
        "action_type": "retry_failed_stale_jobs",
        "reason": f"Retry {limit} failed jobs that exhausted stale-lease recovery.",
        "risk": _ACTION_RISK["retry_failed_stale_jobs"],
        "dry_run": True,
        "params": {"all": False, "limit": limit},
        "expected_state": {},
        "resource": {"type": "jobs", "id": "failed_stale"},
    }


def _baseline_next_action(status_payload: dict[str, Any]) -> dict[str, Any] | None:
    scheduler = _nested_dict(status_payload, "campaign_program", "scheduler")
    active_hash = _optional_str(scheduler.get("active_hash"))
    baseline = status_payload.get("baseline")
    baseline_status = (
        _optional_str(baseline.get("root_baseline_status"))
        if isinstance(baseline, dict)
        else None
    )
    if not active_hash or (baseline is not None and baseline_status != "failed"):
        return None
    expected_state: dict[str, Any] = {"campaign_program_hash": active_hash}
    if baseline_status is not None:
        expected_state["baseline_status"] = baseline_status
    return {
        "action_type": "baseline_ensure",
        "reason": "Ensure the active campaign has a valid root baseline.",
        "risk": _ACTION_RISK["baseline_ensure"],
        "dry_run": True,
        "params": {},
        "expected_state": expected_state,
        "resource": {"type": "campaign_program", "id": active_hash},
    }


def _repair_next_action(
    status_payload: dict[str, Any],
    *,
    settings: Settings,
) -> dict[str, Any] | None:
    if not _repair_capacity_available(status_payload, settings=settings):
        return None
    return {
        "action_type": "repair_schedule_one",
        "reason": "Schedule one eligible failed-candidate repair while repair capacity is available.",
        "risk": _ACTION_RISK["repair_schedule_one"],
        "dry_run": True,
        "params": {},
        "expected_state": {},
        "resource": {"type": "repair_pool", "id": "eligible"},
    }


def _repair_capacity_available(
    status_payload: dict[str, Any],
    *,
    settings: Settings,
) -> bool:
    repair_pool = _nested_dict(status_payload, "repair_pool")
    by_repair_state = repair_pool.get("by_repair_state")
    eligible_count = (
        _safe_int(by_repair_state.get("eligible"))
        if isinstance(by_repair_state, dict)
        else 0
    )
    max_active = max(0, int(settings.failed_candidate_repair_max_active_jobs))
    return (
        eligible_count > 0
        and bool(settings.failed_candidate_repair_enabled)
        and max_active > 0
        and _safe_int(repair_pool.get("active_repair_jobs")) < max_active
        and int(settings.failed_candidate_repair_max_jobs_per_tick) > 0
        and int(settings.failed_candidate_repair_max_tokens) > 0
    )


def run_agent_action(
    request: AgentActionRequest,
    *,
    actor: str,
    background_tasks: Any | None = None,
) -> dict[str, Any]:
    """Validate, optionally execute, and persist one agent action audit row."""

    action_type = _normalize_action_type(request.action_type)
    idempotency_key = _normalize_idempotency_key(request.idempotency_key)
    pending_error: AgentAPIError | None = None
    output: dict[str, Any] | None = None
    with session_scope() as session:
        row = _new_pending_action(
            request=request,
            actor=actor,
            action_type=action_type,
            idempotency_key=idempotency_key,
        )
        replayed = _insert_pending_action(
            session=session,
            row=row,
            action_type=action_type,
            idempotency_key=idempotency_key,
        )
        if replayed is not None:
            return replayed
        pending_error = _complete_action(
            session=session,
            row=row,
            request=request,
            action_type=action_type,
            background_tasks=background_tasks,
        )
        session.flush()
        output = _action_record_payload(row)

    if pending_error is not None:
        raise pending_error
    return _require_action_output(output)


def _insert_pending_action(
    *,
    session: object,
    row: AgentAction,
    action_type: str,
    idempotency_key: str,
) -> dict[str, Any] | None:
    existing = _idempotent_action(session, action_type=action_type, key=idempotency_key)
    if existing is not None:
        log.bind(action_id=str(existing.id), action_type=existing.action_type).info(
            "Agent action idempotency replay"
        )
        return _action_record_payload(existing)

    session.add(row)
    try:
        session.flush()
    except IntegrityError as exc:
        return _idempotency_replay_after_insert_race(
            session=session,
            action_type=action_type,
            idempotency_key=idempotency_key,
            error=exc,
        )
    return None


def _idempotency_replay_after_insert_race(
    *,
    session: object,
    action_type: str,
    idempotency_key: str,
    error: IntegrityError,
) -> dict[str, Any] | None:
    if idempotency_key:
        session.rollback()
        replayed = _load_idempotent_action(action_type=action_type, key=idempotency_key)
        if replayed is not None:
            log.bind(action_id=str(replayed["action_id"]), action_type=action_type).info(
                "Agent action idempotency replay after insert race"
            )
            return replayed
    raise AgentAPIError(
        status_code=500,
        error_code="internal_error",
        message="Agent action audit row could not be created.",
        retryable=True,
    ) from error


def _complete_action(
    *,
    session: object,
    row: AgentAction,
    request: AgentActionRequest,
    action_type: str,
    background_tasks: Any | None,
) -> AgentAPIError | None:
    preconditions: list[dict[str, Any]] = []
    try:
        result, preconditions = _validated_action_result(
            session=session,
            request=request,
            action_type=action_type,
            background_tasks=background_tasks,
        )
        _mark_action_succeeded(row, preconditions=preconditions, result=result)
        log.bind(action_id=str(row.id), action_type=row.action_type, dry_run=row.dry_run).info(
            "Agent action completed"
        )
        return None
    except AgentAPIError as exc:
        preconditions = exc.preconditions or preconditions
        _mark_action_failed(row, error=exc, preconditions=preconditions)
        log.bind(
            action_id=str(row.id),
            action_type=row.action_type,
            error_code=exc.error_code,
        ).warning("Agent action failed")
        return exc
    except Exception as exc:  # pragma: no cover - defensive mapping
        error = _internal_action_error(exc)
        _mark_action_failed(row, error=error, preconditions=preconditions)
        log.bind(action_id=str(row.id), action_type=row.action_type).exception(
            "Agent action failed unexpectedly"
        )
        return error


def _validated_action_result(
    *,
    session: object,
    request: AgentActionRequest,
    action_type: str,
    background_tasks: Any | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    preconditions, context = _validate_action(
        session=session,
        request=request,
        action_type=action_type,
    )
    if request.dry_run:
        return _dry_run_result(action_type=action_type, context=context), preconditions
    return (
        _execute_action(
            request=request,
            action_type=action_type,
            background_tasks=background_tasks,
        ),
        preconditions,
    )


def _internal_action_error(exc: Exception) -> AgentAPIError:
    return AgentAPIError(
        status_code=500,
        error_code="internal_error",
        message=clamp_text(normalize_single_line(str(exc)), 512) or "Agent action failed.",
        retryable=True,
    )


def _require_action_output(output: dict[str, Any] | None) -> dict[str, Any]:
    if output is not None:
        return output
    raise AgentAPIError(
        status_code=500,
        error_code="internal_error",
        message="Agent action did not produce a result.",
        retryable=True,
    )


def get_agent_action_record(*, action_id: UUID) -> dict[str, Any]:
    with session_scope() as session:
        row = session.get(AgentAction, action_id)
        if row is None:
            raise AgentAPIError(
                status_code=404,
                error_code="not_found",
                message="Agent action not found.",
                retryable=False,
                resource={"type": "agent_action", "id": str(action_id)},
            )
        return _action_record_payload(row)


def agent_feedback_payload(*, resource_type: str, resource_id: str, rows: list[object]) -> dict[str, Any]:
    agent_visible_rows = [
        row for row in rows if getattr(row, "visibility", None) == "agent_visible"
    ]
    artifacts = [build_evaluation_artifact_payload(row) for row in agent_visible_rows]
    return {
        "resource": {"type": resource_type, "id": resource_id},
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "feedback": build_agent_feedback_payload(agent_visible_rows),
    }


def _capability_action(
    action_type: str,
    *,
    required_params: list[str] | None = None,
    expected_state_fields: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "action_type": action_type,
        "risk": _ACTION_RISK[action_type],
        "dry_run_supported": True,
        "reason_expected": True,
        "idempotency_key_expected": True,
        "required_params": required_params or [],
        "expected_state_fields": expected_state_fields or [],
    }


def _idempotent_action(
    session: object,
    *,
    action_type: str,
    key: str,
) -> AgentAction | None:
    if not key:
        return None
    return (
        session.execute(
            select(AgentAction)
            .where(
                AgentAction.action_type == action_type,
                AgentAction.idempotency_key == key,
            )
            .order_by(AgentAction.created_at.asc(), AgentAction.id.asc())
            .limit(1)
        )
        .scalars()
        .first()
    )


def _load_idempotent_action(*, action_type: str, key: str) -> dict[str, Any] | None:
    with session_scope() as session:
        row = _idempotent_action(session, action_type=action_type, key=key)
        return _action_record_payload(row) if row is not None else None


def _new_pending_action(
    *,
    request: AgentActionRequest,
    actor: str,
    action_type: str,
    idempotency_key: str,
) -> AgentAction:
    return AgentAction(
        id=uuid.uuid4(),
        idempotency_key=idempotency_key,
        actor=_clean_actor(actor),
        action_type=action_type,
        status="pending",
        dry_run=bool(request.dry_run),
        request_payload=jsonable_encoder(request.model_dump(mode="json")),
        expected_state=jsonable_encoder(request.expected_state),
        result_payload={},
        created_at=_utc_now(),
        completed_at=None,
    )


def _validate_action(
    *,
    session: object,
    request: AgentActionRequest,
    action_type: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if action_type == "retry_job":
        return _validate_retry_job(session=session, request=request)
    if action_type == "retry_failed_stale_jobs":
        return _validate_retry_failed_stale_jobs(session=session, request=request)
    if action_type == "baseline_ensure":
        return _validate_baseline_ensure(request=request)
    if action_type == "repair_schedule_one":
        return _validate_repair_schedule_one()
    if action_type in _CANDIDATE_ACTIONS:
        return _validate_repair_candidate_action(session=session, request=request)
    raise AssertionError(f"unhandled action_type: {action_type}")


def _validate_retry_job(
    *,
    session: object,
    request: AgentActionRequest,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    job_id = _uuid_param(request.params, "job_id", resource_type="job")
    job = session.get(EvolutionJob, job_id)
    if job is None:
        raise AgentAPIError(
            status_code=404,
            error_code="not_found",
            message="Job not found.",
            retryable=False,
            resource={"type": "job", "id": str(job_id)},
        )
    now = _utc_now()
    status = job_status_value(getattr(job, "status", None))
    lease_state = str(job_lease_payload(job=job, now=now).get("state") or "")
    recovery_count = int(getattr(job, "recovery_count", 0) or 0)
    current_state = {
        "status": status,
        "lease_state": lease_state,
        "recovery_count": recovery_count,
    }
    preconditions = _check_expected_state(
        expected=request.expected_state,
        actual=current_state,
        fields=("status", "lease_state", "recovery_count"),
        resource={"type": "job", "id": str(job_id)},
    )
    retryable, retry_lease_state = job_retry_state(job=job, now=now)
    if not retryable:
        raise AgentAPIError(
            status_code=409,
            error_code="conflict",
            message=(
                "Only failed or stuck RUNNING jobs can be retried "
                f"(status={status}, lease_state={retry_lease_state or 'n/a'})."
            ),
            retryable=False,
            resource={"type": "job", "id": str(job_id)},
            preconditions=preconditions,
        )
    return preconditions, {"job_id": str(job_id), "current_state": current_state}


def _validate_retry_failed_stale_jobs(
    *,
    session: object,
    request: AgentActionRequest,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    retry_all, limit = _retry_failed_stale_params(request.params)
    settings = get_settings()
    rows = load_failed_stale_retry_rows(
        session=session,
        max_recovery_attempts=int(settings.scheduler_stale_running_max_recovery_attempts),
        retry_all=retry_all,
        limit=limit,
    )
    return [], {
        "eligible_count": len(rows),
        "retry_all": retry_all,
        "limit": limit,
        "job_ids": [str(getattr(row, "id", "")) for row in rows],
    }


def _validate_baseline_ensure(
    *,
    request: AgentActionRequest,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    status_payload = operator_status()
    scheduler = _nested_dict(status_payload, "campaign_program", "scheduler")
    baseline = status_payload.get("baseline")
    actual = {
        "campaign_program_hash": _optional_str(scheduler.get("active_hash")),
        "baseline_status": (
            _optional_str(baseline.get("root_baseline_status"))
            if isinstance(baseline, dict)
            else None
        ),
        "root_baseline_status": (
            _optional_str(baseline.get("root_baseline_status"))
            if isinstance(baseline, dict)
            else None
        ),
    }
    preconditions = _check_expected_state(
        expected=request.expected_state,
        actual=actual,
        fields=("campaign_program_hash", "baseline_status", "root_baseline_status"),
        resource={
            "type": "campaign_program",
            "id": str(actual["campaign_program_hash"] or "active"),
        },
    )
    return preconditions, {"current_state": actual}


def _validate_repair_schedule_one() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = get_settings()
    return [], {
        "repair_enabled": bool(settings.failed_candidate_repair_enabled),
        "max_active_jobs": int(settings.failed_candidate_repair_max_active_jobs),
        "max_jobs_per_tick": int(settings.failed_candidate_repair_max_jobs_per_tick),
        "max_tokens": int(settings.failed_candidate_repair_max_tokens),
    }


def _validate_repair_candidate_action(
    *,
    session: object,
    request: AgentActionRequest,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_id = _uuid_param(request.params, "candidate_id", resource_type="repair_candidate")
    candidate = session.get(CandidateCommit, candidate_id)
    if candidate is None or normalize_single_line(str(candidate.evaluation_status or "")).lower() != "candidate_failed":
        raise AgentAPIError(
            status_code=404,
            error_code="not_found",
            message="Repair candidate not found.",
            retryable=False,
            resource={"type": "repair_candidate", "id": str(candidate_id)},
        )
    active_job_id = _active_repair_job_id(session=session, candidate_id=candidate_id)
    current_state = {
        "lifecycle_status": _optional_str(candidate.lifecycle_status),
        "repair_state": _optional_str(candidate.repair_state),
        "active_repair_job_id": str(active_job_id) if active_job_id is not None else None,
    }
    preconditions = _check_expected_state(
        expected=request.expected_state,
        actual=current_state,
        fields=("lifecycle_status", "repair_state", "active_repair_job_id"),
        resource={"type": "repair_candidate", "id": str(candidate_id)},
    )
    if active_job_id is not None:
        raise AgentAPIError(
            status_code=409,
            error_code="conflict",
            message="Candidate has an active repair job.",
            retryable=False,
            resource={"type": "repair_candidate", "id": str(candidate_id)},
            preconditions=preconditions,
        )
    return preconditions, {"candidate_id": str(candidate_id), "current_state": current_state}


def _execute_action(
    *,
    request: AgentActionRequest,
    action_type: str,
    background_tasks: Any | None,
) -> dict[str, Any]:
    try:
        return _execute_valid_action(
            request=request,
            action_type=action_type,
            background_tasks=background_tasks,
        )
    except JobNotFoundError as exc:
        raise _mapped_error(exc, status_code=404, error_code="not_found", resource_type="job") from exc
    except (JobRetryConflictError, RepairConflictError, OperatorTaskAlreadyActiveError) as exc:
        raise _mapped_error(exc, status_code=409, error_code="conflict") from exc
    except (JobRetryValidationError, RepairValidationError, ValueError) as exc:
        raise _mapped_error(exc, status_code=400, error_code="invalid_request") from exc
    except RepairNotFoundError as exc:
        raise _mapped_error(
            exc,
            status_code=404,
            error_code="not_found",
            resource_type="repair_candidate",
        ) from exc


def _execute_valid_action(
    *,
    request: AgentActionRequest,
    action_type: str,
    background_tasks: Any | None,
) -> dict[str, Any]:
    if action_type == "retry_job":
        return _execute_retry_job(request)
    if action_type == "retry_failed_stale_jobs":
        return _execute_retry_failed_stale_jobs(request)
    if action_type == "baseline_ensure":
        return _execute_baseline_ensure(background_tasks)
    if action_type == "repair_schedule_one":
        return schedule_one_repair()
    if action_type in _CANDIDATE_ACTIONS:
        return _execute_repair_candidate_action(request, action_type=action_type)
    raise AssertionError(f"unhandled action_type: {action_type}")


def _execute_retry_job(request: AgentActionRequest) -> dict[str, Any]:
    job_id = _uuid_param(request.params, "job_id", resource_type="job")
    return retry_job_by_id(
        job_id=job_id,
        reason=normalize_single_line(str(request.reason or "")),
    )


def _execute_retry_failed_stale_jobs(request: AgentActionRequest) -> dict[str, Any]:
    retry_all, limit = _retry_failed_stale_params(request.params)
    return retry_failed_stale_jobs(
        retry_all=retry_all,
        limit=limit,
        reason=normalize_single_line(str(request.reason or "")),
    )


def _execute_baseline_ensure(background_tasks: Any | None) -> dict[str, Any]:
    task = create_baseline_ensure_task()
    if background_tasks is not None:
        background_tasks.add_task(run_baseline_ensure_task, task.id)
    return {
        "operator_task_id": str(task.id),
        "operator_task_status": str(task.status),
        "background_task_enqueued": background_tasks is not None,
    }


def _execute_repair_candidate_action(
    request: AgentActionRequest,
    *,
    action_type: str,
) -> dict[str, Any]:
    candidate_id = _uuid_param(
        request.params,
        "candidate_id",
        resource_type="repair_candidate",
    )
    return update_candidate_operator_state(
        candidate_id=candidate_id,
        action=_CANDIDATE_ACTIONS[action_type],
    )


def _dry_run_result(*, action_type: str, context: dict[str, Any]) -> dict[str, Any]:
    return {
        "validated": True,
        "would_execute": action_type,
        **jsonable_encoder(context),
    }


def _retry_failed_stale_params(params: dict[str, Any]) -> tuple[bool, int | None]:
    retry_all = _retry_all_param(params)
    limit_value = params.get("limit")
    limit = None
    if limit_value is not None:
        try:
            limit = int(limit_value)
        except (TypeError, ValueError) as exc:
            raise AgentAPIError(
                status_code=400,
                error_code="invalid_request",
                message="retry_failed_stale_jobs limit must be an integer.",
                retryable=False,
                resource={"type": "jobs", "id": "failed_stale"},
            ) from exc
        if limit < 1:
            raise AgentAPIError(
                status_code=400,
                error_code="invalid_request",
                message="retry_failed_stale_jobs limit must be at least 1.",
                retryable=False,
                resource={"type": "jobs", "id": "failed_stale"},
            )
    if retry_all and limit is not None:
        raise AgentAPIError(
            status_code=400,
            error_code="invalid_request",
            message="Use either all=true or limit, not both.",
            retryable=False,
            resource={"type": "jobs", "id": "failed_stale"},
        )
    if not retry_all and limit is None:
        raise AgentAPIError(
            status_code=400,
            error_code="invalid_request",
            message="Use either all=true or limit.",
            retryable=False,
            resource={"type": "jobs", "id": "failed_stale"},
        )
    return retry_all, limit


def _retry_all_param(params: dict[str, Any]) -> bool:
    for key in ("all", "retry_all"):
        if key in params:
            return _strict_bool_param(params[key], key=key)
    return False


def _strict_bool_param(value: object, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    raise AgentAPIError(
        status_code=400,
        error_code="invalid_request",
        message=f"retry_failed_stale_jobs {key} must be a boolean.",
        retryable=False,
        resource={"type": "jobs", "id": "failed_stale"},
    )


def _uuid_param(params: dict[str, Any], key: str, *, resource_type: str) -> UUID:
    raw = params.get(key)
    try:
        return UUID(str(raw))
    except (TypeError, ValueError) as exc:
        raise AgentAPIError(
            status_code=400,
            error_code="invalid_request",
            message=f"{key} must be a UUID.",
            retryable=False,
            resource={"type": resource_type, "id": str(raw or "")},
        ) from exc


def _check_expected_state(
    *,
    expected: dict[str, Any],
    actual: dict[str, Any],
    fields: tuple[str, ...],
    resource: dict[str, str],
) -> list[dict[str, Any]]:
    preconditions: list[dict[str, Any]] = []
    for field in fields:
        if field not in expected:
            continue
        expected_value = expected.get(field)
        actual_value = actual.get(field)
        passed = _state_values_equal(expected_value, actual_value)
        preconditions.append(
            {
                "name": field,
                "expected": expected_value,
                "actual": actual_value,
                "passed": passed,
            }
        )
        if not passed:
            raise AgentAPIError(
                status_code=409,
                error_code="precondition_failed",
                message=f"Expected state mismatch for {field}.",
                retryable=False,
                resource=resource,
                suggested_next_actions=[],
                preconditions=preconditions,
            )
    return preconditions


def _state_values_equal(expected: object, actual: object) -> bool:
    if expected is None or actual is None:
        return expected is None and actual is None
    if isinstance(actual, int):
        try:
            return int(expected) == actual
        except (TypeError, ValueError):
            return False
    return normalize_single_line(str(expected)).lower() == normalize_single_line(str(actual)).lower()


def _active_repair_job_id(*, session: object, candidate_id: UUID) -> UUID | None:
    row = (
        session.execute(
            select(EvolutionJob.id)
            .where(
                EvolutionJob.job_kind == "repair",
                EvolutionJob.repair_source_candidate_id == candidate_id,
                EvolutionJob.status.in_(_ACTIVE_JOB_STATUSES),
            )
            .order_by(EvolutionJob.created_at.desc(), EvolutionJob.id.desc())
            .limit(1)
        )
        .scalars()
        .first()
    )
    return row


def _mark_action_succeeded(
    row: AgentAction,
    *,
    preconditions: list[dict[str, Any]],
    result: dict[str, Any],
) -> None:
    row.status = _ACTION_STATUS_SUCCEEDED
    row.result_payload = {
        "preconditions": jsonable_encoder(preconditions),
        "result": jsonable_encoder(result),
    }
    row.error_code = None
    row.error_summary = None
    row.completed_at = _utc_now()


def _mark_action_failed(
    row: AgentAction,
    *,
    error: AgentAPIError,
    preconditions: list[dict[str, Any]],
) -> None:
    row.status = _ACTION_STATUS_FAILED
    row.result_payload = {
        "preconditions": jsonable_encoder(preconditions),
        "result": None,
    }
    row.error_code = error.error_code
    row.error_summary = clamp_text(normalize_single_line(error.message), 2048)
    row.completed_at = _utc_now()


def _action_record_payload(row: AgentAction) -> dict[str, Any]:
    result_payload = row.result_payload if isinstance(row.result_payload, dict) else {}
    preconditions = result_payload.get("preconditions")
    if not isinstance(preconditions, list):
        preconditions = []
    error = None
    if row.error_code:
        error = {
            "error_code": row.error_code,
            "message": row.error_summary or row.error_code,
            "retryable": row.error_code in {"internal_error"},
            "resource": {"type": "agent_action", "id": str(row.id)},
            "suggested_next_actions": [],
        }
    return {
        "action_id": row.id,
        "status": row.status,
        "dry_run": bool(row.dry_run),
        "action_type": row.action_type,
        "risk": _ACTION_RISK.get(row.action_type, "unknown"),
        "preconditions": preconditions,
        "result": result_payload.get("result"),
        "error": error,
        "created_at": row.created_at,
        "completed_at": row.completed_at,
    }


def _mapped_error(
    exc: Exception,
    *,
    status_code: int,
    error_code: str,
    resource_type: str | None = None,
) -> AgentAPIError:
    resource = {"type": resource_type, "id": ""} if resource_type else None
    return AgentAPIError(
        status_code=status_code,
        error_code=error_code,
        message=clamp_text(normalize_single_line(str(exc)), 512) or error_code,
        retryable=False,
        resource=resource,
    )


def _normalize_action_type(value: object) -> str:
    action_type = normalize_single_line(str(value or ""))
    if not action_type:
        raise AgentAPIError(
            status_code=400,
            error_code="invalid_action_type",
            message="Unsupported agent action_type: <empty>.",
            retryable=False,
            resource={"type": "agent_action", "id": "unknown"},
        )
    if len(action_type) > _MAX_ACTION_TYPE_CHARS:
        raise AgentAPIError(
            status_code=400,
            error_code="invalid_action_type",
            message=f"agent action_type must be at most {_MAX_ACTION_TYPE_CHARS} characters.",
            retryable=False,
            resource={"type": "agent_action", "id": "unknown"},
        )
    if action_type not in _ACTION_RISK:
        raise AgentAPIError(
            status_code=400,
            error_code="invalid_action_type",
            message=f"Unsupported agent action_type: {action_type}.",
            retryable=False,
            resource={"type": "agent_action", "id": action_type},
        )
    return action_type


def _normalize_idempotency_key(value: object) -> str:
    return clamp_text(normalize_single_line(str(value or "")), 256)


def _clean_actor(value: object) -> str:
    return clamp_text(normalize_single_line(str(value or "")), 128) or "agent"


def _nested_dict(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    current: object = payload
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def _optional_str(value: object) -> str | None:
    normalized = normalize_single_line(str(value or ""))
    return normalized or None


def _safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

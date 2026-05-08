"""Repair pool endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from loreley.api.pagination import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT, PaginationCursorError
from loreley.api.schemas.repair import (
    RepairActionRequest,
    RepairCandidateActionOut,
    RepairPoolCandidateOut,
    RepairPoolPageOut,
    RepairScheduleOut,
)
from loreley.api.services.repair import (
    RepairConflictError,
    RepairNotFoundError,
    RepairValidationError,
    list_repair_pool_page,
    schedule_one_repair,
    update_candidate_operator_state,
)

router = APIRouter()


@router.get("/repair/pool", response_model=RepairPoolPageOut)
def get_repair_pool(
    repair_state: str | None = Query(default=None),
    lifecycle_status: str | None = Query(default=None),
    failure_kind: str | None = Query(default=None),
    campaign_program_hash: str | None = Query(default=None),
    limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    cursor: str | None = Query(default=None, description="Opaque pagination cursor."),
) -> RepairPoolPageOut:
    try:
        page = list_repair_pool_page(
            repair_state=repair_state,
            lifecycle_status=lifecycle_status,
            failure_kind=failure_kind,
            campaign_program_hash=campaign_program_hash,
            limit=limit,
            cursor=cursor,
        )
    except PaginationCursorError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RepairPoolPageOut(
        items=[RepairPoolCandidateOut.model_validate(item) for item in page.items],
        next_cursor=page.next_cursor,
        summary=page.summary,
    )


@router.post("/repair/schedule-one", response_model=RepairScheduleOut)
def post_repair_schedule_one() -> RepairScheduleOut:
    try:
        return RepairScheduleOut.model_validate(schedule_one_repair())
    except RepairValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/repair/candidates/{candidate_id}/quarantine", response_model=RepairCandidateActionOut)
def quarantine_repair_candidate(
    candidate_id: UUID,
    _body: RepairActionRequest | None = None,
) -> RepairCandidateActionOut:
    return _candidate_action(candidate_id=candidate_id, action="quarantine")


@router.post("/repair/candidates/{candidate_id}/discard", response_model=RepairCandidateActionOut)
def discard_repair_candidate(
    candidate_id: UUID,
    _body: RepairActionRequest | None = None,
) -> RepairCandidateActionOut:
    return _candidate_action(candidate_id=candidate_id, action="discard")


@router.post("/repair/candidates/{candidate_id}/restore", response_model=RepairCandidateActionOut)
def restore_repair_candidate(
    candidate_id: UUID,
    _body: RepairActionRequest | None = None,
) -> RepairCandidateActionOut:
    return _candidate_action(candidate_id=candidate_id, action="restore")


def _candidate_action(*, candidate_id: UUID, action: str) -> RepairCandidateActionOut:
    try:
        candidate = update_candidate_operator_state(candidate_id=candidate_id, action=action)
    except RepairNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RepairConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RepairValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RepairCandidateActionOut(
        candidate=RepairPoolCandidateOut.model_validate(candidate),
        action=action,
    )

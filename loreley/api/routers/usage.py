"""LLM usage endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from loreley.api.pagination import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT, PaginationCursorError
from loreley.api.schemas.usage import UsageEventOut, UsageEventPageOut, UsageSummaryOut
from loreley.api.services.usage import list_usage_events_page, usage_summary

router = APIRouter()


@router.get("/usage/summary", response_model=UsageSummaryOut)
def get_usage_summary(
    job_id: UUID | None = None,
    source: str | None = Query(default=None),
    phase: str | None = Query(default=None),
    model: str | None = Query(default=None),
) -> UsageSummaryOut:
    return UsageSummaryOut.model_validate(
        usage_summary(job_id=job_id, source=source, phase=phase, model=model)
    )


@router.get("/usage/events/page", response_model=UsageEventPageOut)
def get_usage_events_page(
    job_id: UUID | None = None,
    source: str | None = Query(default=None),
    phase: str | None = Query(default=None),
    model: str | None = Query(default=None),
    limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    cursor: str | None = Query(default=None, description="Opaque pagination cursor."),
) -> UsageEventPageOut:
    try:
        page = list_usage_events_page(
            job_id=job_id,
            source=source,
            phase=phase,
            model=model,
            limit=limit,
            cursor=cursor,
        )
    except PaginationCursorError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return UsageEventPageOut(
        items=[UsageEventOut.model_validate(row) for row in page.items],
        next_cursor=page.next_cursor,
    )

"""LLM usage queries for the UI API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID

from sqlalchemy import case, desc, func, or_, select

from loreley.api.pagination import PaginationCursorError, decode_cursor, encode_cursor, normalize_pagination
from loreley.db.base import session_scope
from loreley.db.models import EvolutionJob, LLMUsageEvent


@dataclass(frozen=True, slots=True)
class UsageEventPage:
    items: list[LLMUsageEvent]
    next_cursor: str | None


class UsageJobNotFoundError(RuntimeError):
    """Raised when job usage is requested for an unknown job."""


def list_usage_events_page(
    *,
    limit: int,
    cursor: str | None = None,
    job_id: UUID | None = None,
    source: str | None = None,
    phase: str | None = None,
    model: str | None = None,
) -> UsageEventPage:
    limit_i, _ = normalize_pagination(limit, 0)
    stmt = _usage_filter_stmt(
        job_id=job_id,
        source=source,
        phase=phase,
        model=model,
    )
    if cursor:
        sort_ts, usage_id = _decode_usage_cursor(cursor)
        stmt = stmt.where(
            or_(
                LLMUsageEvent.created_at < sort_ts,
                (LLMUsageEvent.created_at == sort_ts) & (LLMUsageEvent.id < usage_id),
            )
        )
    stmt = stmt.order_by(desc(LLMUsageEvent.created_at), desc(LLMUsageEvent.id)).limit(limit_i + 1)
    with session_scope() as session:
        rows = list(session.execute(stmt).scalars())
    next_cursor = None
    if len(rows) > limit_i:
        rows = rows[:limit_i]
        next_cursor = _encode_usage_cursor(rows[-1])
    return UsageEventPage(items=rows, next_cursor=next_cursor)


def list_usage_events_for_job(*, job_id: UUID) -> list[LLMUsageEvent]:
    with session_scope() as session:
        if session.get(EvolutionJob, job_id) is None:
            raise UsageJobNotFoundError(f"Job not found: {job_id}")
        return list(
            session.execute(
                select(LLMUsageEvent)
                .where(LLMUsageEvent.job_id == job_id)
                .order_by(LLMUsageEvent.created_at.asc(), LLMUsageEvent.id.asc())
            ).scalars()
        )


def usage_summary(
    *,
    job_id: UUID | None = None,
    source: str | None = None,
    phase: str | None = None,
    model: str | None = None,
) -> dict[str, object]:
    with session_scope() as session:
        base = _usage_filter_stmt(job_id=job_id, source=source, phase=phase, model=model)
        total = _summary_row(session, base.subquery())
        return {
            **total,
            "by_source": _summary_groups(session, base.subquery(), "source"),
            "by_phase": _summary_groups(session, base.subquery(), "phase"),
            "by_model": _summary_groups(session, base.subquery(), "model"),
        }


def _usage_filter_stmt(
    *,
    job_id: UUID | None,
    source: str | None,
    phase: str | None,
    model: str | None,
):
    stmt = select(LLMUsageEvent)
    if job_id is not None:
        stmt = stmt.where(LLMUsageEvent.job_id == job_id)
    if source:
        stmt = stmt.where(LLMUsageEvent.source == source)
    if phase:
        stmt = stmt.where(LLMUsageEvent.phase == phase)
    if model:
        stmt = stmt.where(LLMUsageEvent.model == model)
    return stmt


def _summary_row(session, subquery) -> dict[str, object]:
    row = session.execute(
        select(
            func.count().label("event_count"),
            func.coalesce(func.sum(subquery.c.total_tokens), 0).label("total_tokens"),
            func.coalesce(func.sum(subquery.c.input_tokens), 0).label("input_tokens"),
            func.coalesce(func.sum(subquery.c.cached_input_tokens), 0).label("cached_input_tokens"),
            func.coalesce(func.sum(subquery.c.cache_write_tokens), 0).label("cache_write_tokens"),
            func.coalesce(func.sum(subquery.c.output_tokens), 0).label("output_tokens"),
            func.coalesce(func.sum(subquery.c.reasoning_output_tokens), 0).label("reasoning_output_tokens"),
            func.sum(subquery.c.cost_usd).label("cost_usd"),
            func.coalesce(
                func.sum(case((subquery.c.cost_source == "unpriced", 1), else_=0)),
                0,
            ).label("unpriced_events"),
            func.coalesce(
                func.sum(case((subquery.c.cost_source == "unavailable", 1), else_=0)),
                0,
            ).label("unavailable_events"),
        )
    ).mappings().one()
    return _coerce_summary_mapping(row)


def _summary_groups(session, subquery, key_name: str) -> list[dict[str, object]]:
    key_column = getattr(subquery.c, key_name)
    rows = (
        session.execute(
            select(
                key_column.label("key"),
                func.count().label("event_count"),
                func.coalesce(func.sum(subquery.c.total_tokens), 0).label("total_tokens"),
                func.coalesce(func.sum(subquery.c.input_tokens), 0).label("input_tokens"),
                func.coalesce(func.sum(subquery.c.cached_input_tokens), 0).label("cached_input_tokens"),
                func.coalesce(func.sum(subquery.c.cache_write_tokens), 0).label("cache_write_tokens"),
                func.coalesce(func.sum(subquery.c.output_tokens), 0).label("output_tokens"),
                func.coalesce(func.sum(subquery.c.reasoning_output_tokens), 0).label("reasoning_output_tokens"),
                func.sum(subquery.c.cost_usd).label("cost_usd"),
                func.coalesce(
                    func.sum(case((subquery.c.cost_source == "unpriced", 1), else_=0)),
                    0,
                ).label("unpriced_events"),
                func.coalesce(
                    func.sum(case((subquery.c.cost_source == "unavailable", 1), else_=0)),
                    0,
                ).label("unavailable_events"),
            )
            .group_by(key_column)
            .order_by(desc(func.coalesce(func.sum(subquery.c.total_tokens), 0)), key_column.asc())
        )
        .mappings()
        .all()
    )
    return [_coerce_summary_mapping(row, key_name="key") for row in rows]


def _coerce_summary_mapping(row, *, key_name: str | None = None) -> dict[str, object]:
    payload = dict(row)
    if key_name:
        payload[key_name] = str(payload.get(key_name) or "")
    for name in (
        "event_count",
        "total_tokens",
        "input_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "unpriced_events",
        "unavailable_events",
    ):
        payload[name] = int(payload.get(name) or 0)
    cost = payload.get("cost_usd")
    payload["cost_usd"] = str(cost) if isinstance(cost, Decimal) else (str(cost) if cost is not None else None)
    return payload


def _encode_usage_cursor(event: LLMUsageEvent) -> str:
    return encode_cursor(
        {
            "sort_ts": event.created_at.isoformat(),
            "usage_id": str(event.id),
        }
    )


def _decode_usage_cursor(cursor: str) -> tuple[datetime, UUID]:
    payload = decode_cursor(cursor)
    sort_raw = payload.get("sort_ts")
    usage_raw = payload.get("usage_id")
    if not isinstance(sort_raw, str) or not isinstance(usage_raw, str):
        raise PaginationCursorError("Usage cursor is missing required fields.")
    try:
        sort_ts = datetime.fromisoformat(sort_raw)
    except ValueError as exc:
        raise PaginationCursorError("Usage cursor has an invalid timestamp.") from exc
    if sort_ts.tzinfo is None:
        sort_ts = sort_ts.replace(tzinfo=timezone.utc)
    try:
        usage_id = UUID(usage_raw)
    except ValueError as exc:
        raise PaginationCursorError("Usage cursor has an invalid id.") from exc
    return sort_ts, usage_id

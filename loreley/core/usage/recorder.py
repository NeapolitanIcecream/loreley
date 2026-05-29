from __future__ import annotations

from typing import Any, Iterable

from loguru import logger
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from loreley.config import Settings, get_settings
from loreley.core.usage.events import LLMUsageEventPayload

log = logger.bind(module="usage.recorder")


def record_usage_event(
    event: LLMUsageEventPayload | None,
    *,
    settings: Settings | None = None,
) -> int:
    if event is None:
        return 0
    try:
        return persist_usage_events((event,), settings=settings)
    except Exception as exc:  # pragma: no cover - defensive best effort
        log.warning("Failed to record LLM usage event source={} phase={}: {}", event.source, event.phase, exc)
        return 0


def persist_usage_events(
    events: Iterable[LLMUsageEventPayload | None],
    *,
    session: Any | None = None,
    settings: Settings | None = None,
) -> int:
    settings = settings or get_settings()
    if not bool(getattr(settings, "llm_usage_tracking_enabled", True)):
        return 0
    materialized = tuple(event for event in events if event is not None)
    if not materialized:
        return 0
    if session is not None:
        return _persist_with_session(session, materialized)

    from loreley.db.base import session_scope

    try:
        with session_scope() as scoped_session:
            return _persist_with_session(scoped_session, materialized)
    except SQLAlchemyError as exc:
        log.warning("Failed to persist LLM usage events: {}", exc)
        return 0


def _persist_with_session(session: Any, events: tuple[LLMUsageEventPayload, ...]) -> int:
    from loreley.db.models import LLMUsageEvent

    inserted = 0
    for event in events:
        external_id = str(event.external_usage_id or "").strip()
        if external_id:
            existing = session.execute(
                select(LLMUsageEvent.id).where(
                    LLMUsageEvent.external_usage_id == external_id,
                )
            ).scalar_one_or_none()
            if existing is not None:
                continue
        session.add(LLMUsageEvent(**event.as_record_dict()))
        inserted += 1
    if inserted:
        log.debug("Persisted {} LLM usage event(s)", inserted)
    return inserted

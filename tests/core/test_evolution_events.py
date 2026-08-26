from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from loreley.core.evolution_events import (
    CODING_STARTED,
    EvolutionEventValidationError,
    evolution_event_key,
    finish_evolution_stage,
    record_evolution_event,
    sanitized_event_payload,
    start_evolution_stage,
)
from loreley.db.models import EvolutionEvent


class _RecordingSession:
    def __init__(self) -> None:
        self.added: list[object] = []

    def add(self, value: object) -> None:
        self.added.append(value)


def test_event_key_is_deterministic_and_distinguishes_ordinals() -> None:
    job_id = uuid.uuid4()
    run_token = uuid.uuid4()

    first = evolution_event_key(
        CODING_STARTED,
        job_id=job_id,
        run_token=run_token,
        ordinal=1,
        key_parts=("stage", "coding"),
    )
    replay = evolution_event_key(
        CODING_STARTED,
        job_id=job_id,
        run_token=run_token,
        ordinal=1,
        key_parts=("stage", "coding"),
    )
    second = evolution_event_key(
        CODING_STARTED,
        job_id=job_id,
        run_token=run_token,
        ordinal=2,
        key_parts=("stage", "coding"),
    )

    assert first == replay
    assert first != second
    assert first.startswith("evo1:")


def test_record_event_rejects_unknown_or_unbounded_payload() -> None:
    session = _RecordingSession()

    with pytest.raises(EvolutionEventValidationError, match="payload fields"):
        record_evolution_event(
            session,
            event_type=CODING_STARTED,
            payload={"raw_prompt": "secret"},
        )
    with pytest.raises(EvolutionEventValidationError, match="payload sequences"):
        sanitized_event_payload(
            CODING_STARTED,
            {"rework": list(range(33))},
        )
    assert session.added == []


def test_stage_start_and_finish_use_distinct_events_and_monotonic_duration() -> None:
    session = _RecordingSession()
    job_id = uuid.uuid4()
    run_token = uuid.uuid4()
    started_at = datetime(2026, 8, 25, 10, 0, tzinfo=UTC)

    handle = start_evolution_stage(
        session,
        stage="coding",
        job_id=job_id,
        run_token=run_token,
        ordinal=2,
        occurred_at=started_at,
        payload={"rework": True},
    )
    receipt = finish_evolution_stage(
        session,
        handle=handle,
        outcome="succeeded",
        payload={"rework": True},
    )

    rows = [row for row in session.added if isinstance(row, EvolutionEvent)]
    assert [row.event_type for row in rows] == [
        "coding.started",
        "coding.finished",
    ]
    assert rows[0].occurred_at == started_at
    assert rows[0].ordinal == rows[1].ordinal == 2
    assert rows[1].duration_seconds is not None
    assert rows[1].duration_seconds >= 0
    assert receipt.event_key != rows[0].event_key

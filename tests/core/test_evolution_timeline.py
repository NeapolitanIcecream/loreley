from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from loreley.core.evolution_timeline import (
    EvolutionTimelineExport,
    _archive_membership_issues,
    _derive_interrupted_stages,
    _job_terminal_issues,
    _stage_pairing_issues,
)


def _event(
    event_type: str,
    *,
    job_id: str,
    run_token: str | None,
    ordinal: int,
    occurred_at: datetime,
    event_id: str | None = None,
    island_id: str | None = None,
    commit_hash: str | None = None,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "record_type": "event",
        "timeline_schema_version": 1,
        "event_id": event_id or f"{event_type}:{ordinal}",
        "source": "test",
        "event_type": event_type,
        "occurred_at": occurred_at.isoformat().replace("+00:00", "Z"),
        "job_id": job_id,
        "run_token": run_token,
        "island_id": island_id,
        "commit_hash": commit_hash,
        "ordinal": ordinal,
        "duration_seconds": None,
        "payload": payload or {},
    }


def test_jsonl_export_is_byte_deterministic() -> None:
    metadata = {
        "record_type": "metadata",
        "timeline_schema_version": 1,
        "strict_valid": True,
    }
    event = _event(
        "planning.started",
        job_id=str(uuid.uuid4()),
        run_token=str(uuid.uuid4()),
        ordinal=1,
        occurred_at=datetime(2026, 8, 25, 1, 2, 3, tzinfo=UTC),
    )
    exported = EvolutionTimelineExport(
        metadata=metadata,
        events=(event,),
        issues=(),
    )

    assert exported.to_jsonl() == exported.to_jsonl()
    assert "prompt" not in exported.to_jsonl().lower()


def test_failed_run_derives_explicit_interrupted_evaluation() -> None:
    job_id = str(uuid.uuid4())
    run_token = str(uuid.uuid4())
    started = datetime(2026, 8, 25, 1, 0, tzinfo=UTC)
    failed = started + timedelta(seconds=12)
    records = [
        _event(
            "evaluation.invocation.started",
            job_id=job_id,
            run_token=run_token,
            ordinal=2,
            occurred_at=started,
        ),
        _event(
            "job.failed",
            job_id=job_id,
            run_token=run_token,
            ordinal=1,
            occurred_at=failed,
        ),
    ]

    derived = _derive_interrupted_stages(records)

    assert len(derived) == 1
    assert derived[0]["event_type"] == "evaluation.interrupted"
    assert derived[0]["duration_seconds"] == 12.0


def test_terminal_run_requires_start_with_the_same_run_token() -> None:
    job_id = str(uuid.uuid4())
    old_run_token = str(uuid.uuid4())
    final_run_token = str(uuid.uuid4())
    now = datetime(2026, 8, 25, 1, 0, tzinfo=UTC)
    records = [
        _event(
            "job.run.started",
            job_id=job_id,
            run_token=old_run_token,
            ordinal=1,
            occurred_at=now,
        ),
        _event(
            "job.succeeded",
            job_id=job_id,
            run_token=final_run_token,
            ordinal=1,
            occurred_at=now + timedelta(seconds=10),
        ),
    ]
    job = SimpleNamespace(
        id=uuid.UUID(job_id),
        status=SimpleNamespace(value="succeeded"),
        created_at=now - timedelta(seconds=1),
    )

    issues = list(_job_terminal_issues(records, jobs=[job]))

    assert [issue.code for issue in issues] == ["terminal_without_run_start"]
    assert issues[0].event_id == "job.succeeded:1"
    assert final_run_token in issues[0].message


def test_strict_pairing_reports_finish_without_start_and_terminal_active_stage() -> (
    None
):
    job_id = str(uuid.uuid4())
    run_token = str(uuid.uuid4())
    now = datetime(2026, 8, 25, 1, 0, tzinfo=UTC)
    records = [
        _event(
            "planning.finished",
            job_id=job_id,
            run_token=run_token,
            ordinal=1,
            occurred_at=now,
        ),
        _event(
            "coding.started",
            job_id=job_id,
            run_token=run_token,
            ordinal=1,
            occurred_at=now,
        ),
    ]
    job = SimpleNamespace(
        id=uuid.UUID(job_id),
        status=SimpleNamespace(value="failed"),
        ingestion_status=None,
    )

    issues = list(_stage_pairing_issues(records, jobs=[job]))

    assert {issue.code for issue in issues} == {
        "stage_finish_without_start",
        "active_stage_without_interruption",
    }


def test_archive_removal_requires_prior_membership_but_boundary_satisfies_it() -> None:
    now = datetime(2026, 8, 25, 1, 0, tzinfo=UTC)
    job_id = str(uuid.uuid4())
    removal = _event(
        "archive.member.removed",
        job_id=job_id,
        run_token=None,
        ordinal=1,
        occurred_at=now,
        island_id="main",
        commit_hash="abc123",
        payload={"from_cell": 4, "reason": "local_pareto_update"},
    )

    broken = list(_archive_membership_issues([removal]))
    initial = _event(
        "archive.member.initial_state",
        job_id=job_id,
        run_token=None,
        ordinal=1,
        occurred_at=now - timedelta(seconds=1),
        island_id="main",
        commit_hash="abc123",
        payload={"cell_index": 4, "reason": "migration_boundary"},
    )
    valid = list(_archive_membership_issues([initial, removal]))

    assert [issue.code for issue in broken] == ["archive_removal_without_membership"]
    assert valid == []

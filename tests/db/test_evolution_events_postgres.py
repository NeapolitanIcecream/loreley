from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter

import pytest
from rich.console import Console
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

import loreley.core.worker.job_store as job_store_module
import loreley.scheduler.ingestion as ingestion_module
import loreley.scheduler.job_scheduler as scheduler_module
from loreley.core.evolution_events import (
    ARCHIVE_CANDIDATE_CONSIDERED,
    JOB_DISPATCHED,
    JOB_RECLAIMED,
    JOB_RUN_STARTED,
    JOB_SUCCEEDED,
    finish_evolution_stage,
    record_evolution_event,
    start_evolution_stage,
)
from loreley.core.evolution_timeline import export_evolution_timeline
from loreley.core.map_elites.objectives import ObjectiveContract, ObjectiveSpec
from loreley.core.map_elites.snapshot import (
    DatabaseSnapshotStore,
    SnapshotElite,
    SnapshotUpdate,
)
from loreley.core.worker.job_store import EvolutionJobStore
from loreley.db.base import INSTANCE_SCHEMA_VERSION
from loreley.db.migrations.runner import ensure_schema_current
from loreley.db.models import (
    EvaluationAttempt,
    EvolutionEvent,
    EvolutionJob,
    JobStatus,
    LLMUsageEvent,
)
from loreley.scheduler.ingestion import JobSnapshot, MapElitesIngestion
from loreley.scheduler.job_scheduler import JobScheduler
from tests.support import TestSettings

POSTGRES_TEST_DSN = os.getenv("LORELEY_TEST_DATABASE_URL") or os.getenv(
    "LORELEY_POSTGRES_TEST_DSN"
)


@pytest.fixture
def evolution_engine() -> Iterator[Engine]:
    if not POSTGRES_TEST_DSN:
        pytest.skip(
            "set LORELEY_TEST_DATABASE_URL or LORELEY_POSTGRES_TEST_DSN "
            "to run evolution event PostgreSQL tests"
        )
    schema_name = f"loreley_evolution_event_test_{uuid.uuid4().hex}"
    admin_engine = create_engine(POSTGRES_TEST_DSN, future=True)
    with admin_engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema_name}"'))
    engine = create_engine(
        POSTGRES_TEST_DSN,
        connect_args={"options": f"-csearch_path={schema_name}"},
        future=True,
    )
    settings = TestSettings(
        EXPERIMENT_ID="evolution-events",
        MAPELITES_EXPERIMENT_ROOT_COMMIT=("a" * 40),
        DB_AUTO_MIGRATE=True,
    )
    ensure_schema_current(
        engine=engine,
        settings=settings,
        target_version=INSTANCE_SCHEMA_VERSION,
        auto_migrate=True,
    )
    try:
        yield engine
    finally:
        engine.dispose()
        with admin_engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE'))
        admin_engine.dispose()


def test_duplicate_event_delivery_is_harmless(evolution_engine: Engine) -> None:
    session_factory = sessionmaker(bind=evolution_engine, future=True)
    job_id = uuid.uuid4()
    run_token = uuid.uuid4()
    with session_factory.begin() as session:
        session.add(
            EvolutionJob(
                id=job_id,
                status=JobStatus.RUNNING,
                base_commit_hash="a" * 40,
                island_id="main",
                run_token=run_token,
            )
        )
    with session_factory.begin() as session:
        first = record_evolution_event(
            session,
            event_type=JOB_RUN_STARTED,
            job_id=job_id,
            run_token=run_token,
            island_id="main",
            payload={"job_kind": "evolution", "recovery_count": 0},
            key_parts=("worker_run",),
        )
        replay = record_evolution_event(
            session,
            event_type=JOB_RUN_STARTED,
            job_id=job_id,
            run_token=run_token,
            island_id="main",
            payload={"job_kind": "evolution", "recovery_count": 0},
            key_parts=("worker_run",),
        )

    with session_factory() as session:
        count = session.execute(
            select(func.count(EvolutionEvent.id)).where(EvolutionEvent.job_id == job_id)
        ).scalar_one()
    assert first.inserted is True
    assert replay.inserted is False
    assert replay.event_id == first.event_id
    assert count == 1


def test_v22_to_v23_records_current_archive_as_migration_boundary(
    evolution_engine: Engine,
) -> None:
    settings = TestSettings(
        EXPERIMENT_ID="evolution-events",
        MAPELITES_EXPERIMENT_ROOT_COMMIT="a" * 40,
        DB_AUTO_MIGRATE=True,
    )
    commit_hash = "f" * 40
    with evolution_engine.begin() as connection:
        connection.execute(text("DROP TABLE evolution_events"))
        connection.execute(
            text("UPDATE instance_metadata SET schema_version = 22 WHERE id = 1")
        )
        connection.execute(
            text("DELETE FROM loreley_schema_migrations WHERE version = 23")
        )
        connection.execute(
            text(
                """
                INSERT INTO map_elites_archive_cells (
                    island_id,
                    cell_index,
                    commit_hash,
                    objective_values,
                    measures,
                    timestamp
                )
                VALUES ('main', 7, :commit_hash, ARRAY[1.0], ARRAY[0.5], 1.0)
                """
            ),
            {"commit_hash": commit_hash},
        )

    result = ensure_schema_current(
        engine=evolution_engine,
        settings=settings,
        target_version=INSTANCE_SCHEMA_VERSION,
        auto_migrate=True,
    )

    with evolution_engine.connect() as connection:
        rows = list(
            connection.execute(
                text(
                    "SELECT event_type, island_id, commit_hash, payload "
                    "FROM evolution_events ORDER BY event_type, commit_hash"
                )
            ).mappings()
        )
    assert result.applied_versions == (23,)
    assert [row["event_type"] for row in rows] == [
        "archive.member.initial_state",
        "timeline.history_boundary",
    ]
    initial = rows[0]
    assert initial["island_id"] == "main"
    assert initial["commit_hash"] == commit_hash
    assert initial["payload"] == {
        "cell_index": 7,
        "reason": "migration_boundary",
    }


def test_strict_export_is_deterministic_for_complete_zero_model_job(
    evolution_engine: Engine,
) -> None:
    session_factory = sessionmaker(bind=evolution_engine, future=True)
    job_id = uuid.uuid4()
    run_token = uuid.uuid4()
    started = datetime(2026, 8, 25, 2, 0, tzinfo=UTC)
    finished = started + timedelta(seconds=5)
    with session_factory.begin() as session:
        session.add(
            EvolutionJob(
                id=job_id,
                status=JobStatus.SUCCEEDED,
                base_commit_hash="a" * 40,
                island_id="main",
                inspiration_commit_hashes=["b" * 40],
                result_commit_hash="c" * 40,
                ingestion_status="succeeded",
                ingestion_attempts=1,
                started_at=started,
                completed_at=finished,
                created_at=started - timedelta(seconds=1),
                updated_at=finished,
            )
        )
        record_evolution_event(
            session,
            event_type=JOB_RUN_STARTED,
            job_id=job_id,
            run_token=run_token,
            island_id="main",
            occurred_at=started,
            payload={"job_kind": "evolution", "recovery_count": 0},
            key_parts=("worker_run",),
        )
        planning = start_evolution_stage(
            session,
            stage="planning",
            job_id=job_id,
            run_token=run_token,
            island_id="main",
            ordinal=1,
            occurred_at=started,
        )
        finish_evolution_stage(
            session,
            handle=planning,
            outcome="succeeded",
            occurred_at=started + timedelta(seconds=1),
        )
        coding = start_evolution_stage(
            session,
            stage="coding",
            job_id=job_id,
            run_token=run_token,
            island_id="main",
            ordinal=1,
            occurred_at=started + timedelta(seconds=1),
            payload={"rework": False},
        )
        finish_evolution_stage(
            session,
            handle=coding,
            outcome="succeeded",
            payload={"rework": False},
            occurred_at=started + timedelta(seconds=2),
        )
        evaluation = start_evolution_stage(
            session,
            stage="evaluation",
            job_id=job_id,
            run_token=run_token,
            island_id="main",
            commit_hash="c" * 40,
            ordinal=1,
            occurred_at=started + timedelta(seconds=2),
            payload={"protocol": "one_shot"},
        )
        session.add(
            EvaluationAttempt(
                job_id=job_id,
                run_token=run_token,
                attempt_ordinal=evaluation.ordinal,
                outcome_kind="passed",
                protocol="one_shot",
                started_at=started + timedelta(seconds=2),
                finished_at=started + timedelta(seconds=4),
            )
        )
        record_evolution_event(
            session,
            event_type=JOB_SUCCEEDED,
            job_id=job_id,
            run_token=run_token,
            island_id="main",
            commit_hash="c" * 40,
            occurred_at=finished,
            payload={"outcome": "succeeded"},
            key_parts=("terminal",),
        )
        ingestion = start_evolution_stage(
            session,
            stage="ingestion",
            job_id=job_id,
            island_id="main",
            commit_hash="c" * 40,
            ordinal=1,
            occurred_at=finished + timedelta(seconds=1),
        )
        finish_evolution_stage(
            session,
            handle=ingestion,
            outcome="admitted",
            payload={
                "reason": "archive_member",
                "status_code": 1,
                "inserted": True,
            },
            occurred_at=finished + timedelta(seconds=2),
        )

    with session_factory() as session:
        first = export_evolution_timeline(session, strict=True)
        second = export_evolution_timeline(session, strict=True)

    assert first.issues == ()
    assert first.to_jsonl() == second.to_jsonl()
    assert "evaluation.invocation.finished" in first.to_jsonl()
    assert "raw_prompt" not in first.to_jsonl()
    assert "api_key" not in first.to_jsonl()


def test_ingestion_retry_after_crash_is_strict_valid(
    evolution_engine: Engine,
) -> None:
    session_factory = sessionmaker(bind=evolution_engine, future=True)
    job_id = uuid.uuid4()
    run_token = uuid.uuid4()
    started = datetime(2026, 8, 25, 2, 30, tzinfo=UTC)
    completed = started + timedelta(seconds=5)
    commit_hash = "d" * 40
    with session_factory.begin() as session:
        session.add(
            EvolutionJob(
                id=job_id,
                status=JobStatus.SUCCEEDED,
                base_commit_hash="a" * 40,
                island_id="main",
                inspiration_commit_hashes=[],
                result_commit_hash=commit_hash,
                ingestion_status="succeeded",
                ingestion_attempts=2,
                started_at=started,
                completed_at=completed,
                created_at=started - timedelta(seconds=1),
                updated_at=completed,
            )
        )
        record_evolution_event(
            session,
            event_type=JOB_RUN_STARTED,
            job_id=job_id,
            run_token=run_token,
            island_id="main",
            occurred_at=started,
            payload={"job_kind": "evolution", "recovery_count": 0},
            key_parts=("worker_run",),
        )
        record_evolution_event(
            session,
            event_type=JOB_SUCCEEDED,
            job_id=job_id,
            run_token=run_token,
            island_id="main",
            commit_hash=commit_hash,
            occurred_at=completed,
            payload={"outcome": "succeeded"},
            key_parts=("terminal",),
        )
        start_evolution_stage(
            session,
            stage="ingestion",
            job_id=job_id,
            island_id="main",
            commit_hash=commit_hash,
            ordinal=1,
            occurred_at=completed + timedelta(seconds=1),
        )
        retry = start_evolution_stage(
            session,
            stage="ingestion",
            job_id=job_id,
            island_id="main",
            commit_hash=commit_hash,
            ordinal=2,
            occurred_at=completed + timedelta(seconds=2),
        )
        finish_evolution_stage(
            session,
            handle=retry,
            outcome="admitted",
            payload={
                "reason": "archive_member",
                "status_code": 1,
                "inserted": True,
            },
            occurred_at=completed + timedelta(seconds=3),
        )

    with session_factory() as session:
        exported = export_evolution_timeline(session, strict=True)

    interruptions = [
        event
        for event in exported.events
        if event["event_type"] == "ingestion.interrupted"
    ]
    assert exported.issues == ()
    assert len(interruptions) == 1
    assert interruptions[0]["ordinal"] == 1
    assert interruptions[0]["duration_seconds"] == 1.0
    assert interruptions[0]["payload"]["reason"] == "superseded_by_retry"


def test_archive_delta_records_admission_refit_move_and_pareto_eviction(
    evolution_engine: Engine,
) -> None:
    session_factory = sessionmaker(bind=evolution_engine, future=True)
    job_id = uuid.uuid4()
    contract = ObjectiveContract((ObjectiveSpec(name="quality", direction="max"),))
    store = DatabaseSnapshotStore()
    elite_a_cell_1 = SnapshotElite(
        cell_index=1,
        commit_hash="a" * 40,
        objective_values=(1.0,),
        measures=(0.1,),
        timestamp=1.0,
    )
    elite_a_cell_2 = SnapshotElite(
        cell_index=2,
        commit_hash="a" * 40,
        objective_values=(1.0,),
        measures=(0.2,),
        timestamp=1.0,
    )
    elite_b = SnapshotElite(
        cell_index=3,
        commit_hash="b" * 40,
        objective_values=(2.0,),
        measures=(0.3,),
        timestamp=2.0,
    )
    elite_c = SnapshotElite(
        cell_index=3,
        commit_hash="c" * 40,
        objective_values=(3.0,),
        measures=(0.3,),
        timestamp=3.0,
    )
    with session_factory.begin() as session:
        session.add(
            EvolutionJob(
                id=job_id,
                status=JobStatus.PENDING,
                base_commit_hash="0" * 40,
                island_id="main",
            )
        )
    with session_factory.begin() as session:
        store.apply_update(
            "main",
            update=SnapshotUpdate(
                objective_contract=contract,
                archive_replace=(elite_a_cell_1,),
                event_key_prefix="archive-cycle-1",
                event_job_id=job_id,
                event_ordinal=1,
                archive_change_reason="local_pareto_update",
            ),
            session=session,
        )
    with session_factory.begin() as session:
        store.apply_update(
            "main",
            update=SnapshotUpdate(
                objective_contract=contract,
                archive_replace=(elite_a_cell_2, elite_b),
                event_key_prefix="archive-cycle-2",
                event_job_id=job_id,
                event_ordinal=2,
                archive_change_reason="projection_rebuild",
                projection_epoch=2,
            ),
            session=session,
        )
    with session_factory.begin() as session:
        store.apply_update(
            "main",
            update=SnapshotUpdate(
                objective_contract=contract,
                archive_replace=(elite_a_cell_2, elite_c),
                event_key_prefix="archive-cycle-3",
                event_job_id=job_id,
                event_ordinal=3,
                archive_change_reason="local_pareto_update",
            ),
            session=session,
        )

    with session_factory() as session:
        rows = list(
            session.execute(
                select(EvolutionEvent)
                .where(EvolutionEvent.event_type.like("archive.%"))
                .order_by(EvolutionEvent.occurred_at, EvolutionEvent.id)
            ).scalars()
        )
        exported = export_evolution_timeline(session, strict=False)

    by_type = [row.event_type for row in rows]
    assert by_type.count("archive.member.admitted") == 3
    assert "archive.member.moved" in by_type
    assert "archive.member.removed" in by_type
    assert "archive.rebuild.completed" in by_type
    removed_b = next(
        row
        for row in rows
        if row.event_type == "archive.member.removed" and row.commit_hash == "b" * 40
    )
    assert removed_b.payload == {
        "from_cell": 3,
        "reason": "local_pareto_update",
    }
    assert not any(issue.code.startswith("archive_") for issue in exported.issues)


def test_reclaim_preserves_old_run_and_new_run_provenance(
    evolution_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_factory = sessionmaker(bind=evolution_engine, future=True)
    job_id = uuid.uuid4()
    old_run = uuid.uuid4()
    now = datetime(2026, 8, 25, 4, 0, tzinfo=UTC)
    settings = TestSettings(
        EXPERIMENT_ID="reclaim-events",
        MAPELITES_EXPERIMENT_ROOT_COMMIT="a" * 40,
    )
    settings.scheduler_stale_running_reclaim_batch_size = 10
    settings.scheduler_stale_running_max_recovery_attempts = 2
    settings.worker_job_lease_ttl_seconds = 60
    with session_factory.begin() as session:
        session.add(
            EvolutionJob(
                id=job_id,
                status=JobStatus.RUNNING,
                base_commit_hash="a" * 40,
                island_id="main",
                run_token=old_run,
                worker_id="old-worker",
                started_at=now - timedelta(minutes=3),
                heartbeat_at=now - timedelta(minutes=2),
                lease_expires_at=now - timedelta(minutes=1),
            )
        )
        record_evolution_event(
            session,
            event_type=JOB_RUN_STARTED,
            job_id=job_id,
            run_token=old_run,
            island_id="main",
            occurred_at=now - timedelta(minutes=3),
            payload={"job_kind": "evolution", "recovery_count": 0},
            key_parts=("worker_run",),
        )

    @contextmanager
    def isolated_scope():  # type: ignore[no-untyped-def]
        with session_factory.begin() as session:
            yield session

    monkeypatch.setattr(scheduler_module, "session_scope", isolated_scope)
    monkeypatch.setattr(job_store_module, "session_scope", isolated_scope)
    scheduler = object.__new__(JobScheduler)
    scheduler.settings = settings
    scheduler.console = Console(quiet=True)

    reclaimed = scheduler.reclaim_stale_running_jobs(now=now)
    assert reclaimed.requeued == 1
    assert scheduler._mark_jobs_queued([job_id]) == [job_id]
    locked = EvolutionJobStore(settings=settings).start_job(job_id)

    with session_factory() as session:
        events = list(
            session.execute(
                select(EvolutionEvent)
                .where(EvolutionEvent.job_id == job_id)
                .order_by(EvolutionEvent.occurred_at, EvolutionEvent.id)
            ).scalars()
        )
    reclaim_event = next(
        event for event in events if event.event_type == "job.reclaimed"
    )
    run_tokens = {
        event.run_token for event in events if event.event_type == "job.run.started"
    }
    assert reclaim_event.run_token == old_run
    assert reclaim_event.payload["outcome"] == "requeued"
    assert locked.run_token != old_run
    assert run_tokens == {old_run, locked.run_token}


def test_ingestion_start_is_durable_before_finish_and_classifies_rejection(
    evolution_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_factory = sessionmaker(bind=evolution_engine, future=True)
    job_id = uuid.uuid4()
    commit_hash = "d" * 40
    now = datetime(2026, 8, 25, 5, 0, tzinfo=UTC)
    settings = TestSettings(
        EXPERIMENT_ID="ingestion-events",
        MAPELITES_EXPERIMENT_ROOT_COMMIT="a" * 40,
    )
    with session_factory.begin() as session:
        session.add(
            EvolutionJob(
                id=job_id,
                status=JobStatus.SUCCEEDED,
                base_commit_hash="a" * 40,
                island_id="main",
                result_commit_hash=commit_hash,
                completed_at=now,
            )
        )

    @contextmanager
    def isolated_scope():  # type: ignore[no-untyped-def]
        with session_factory.begin() as session:
            yield session

    monkeypatch.setattr(ingestion_module, "session_scope", isolated_scope)
    ingestion = MapElitesIngestion(
        settings=settings,
        console=Console(quiet=True),
        repo_root=Path("."),
        repo=object(),  # type: ignore[arg-type]
        manager=object(),  # type: ignore[arg-type]
    )
    snapshot = JobSnapshot(
        job_id=job_id,
        base_commit_hash="a" * 40,
        island_id="main",
        result_commit_hash=commit_hash,
        completed_at=now,
    )
    handle = ingestion._start_ingestion_attempt(snapshot)
    assert handle is not None
    with session_factory() as session:
        starts_before_finish = session.execute(
            select(func.count(EvolutionEvent.id)).where(
                EvolutionEvent.job_id == job_id,
                EvolutionEvent.event_type == "ingestion.started",
            )
        ).scalar_one()
        finishes_before_finish = session.execute(
            select(func.count(EvolutionEvent.id)).where(
                EvolutionEvent.job_id == job_id,
                EvolutionEvent.event_type == "ingestion.finished",
            )
        ).scalar_one()
    assert starts_before_finish == 1
    assert finishes_before_finish == 0

    ingestion._record_ingestion_state(
        snapshot,
        status="skipped",
        status_code=0,
        message="Evaluator-equivalent duplicate identity was not admitted.",
    )

    with session_factory() as session:
        rows = list(
            session.execute(
                select(EvolutionEvent)
                .where(EvolutionEvent.job_id == job_id)
                .order_by(EvolutionEvent.occurred_at, EvolutionEvent.id)
            ).scalars()
        )
    assert [row.event_type for row in rows] == [
        "ingestion.started",
        "archive.candidate.considered",
        "ingestion.finished",
    ] or [row.event_type for row in rows] == [
        "ingestion.started",
        "ingestion.finished",
        "archive.candidate.considered",
    ]
    considered = next(
        row for row in rows if row.event_type == "archive.candidate.considered"
    )
    assert considered.ordinal == handle.ordinal
    assert considered.payload == {
        "outcome": "not_admitted",
        "reason": "duplicate_identity",
    }


def test_database_level_zero_model_four_job_timeline_fixture_is_strict_valid(
    evolution_engine: Engine,
) -> None:
    session_factory = sessionmaker(bind=evolution_engine, future=True)
    base_time = datetime(2026, 8, 25, 6, 0, tzinfo=UTC)
    fixture_started = perf_counter()
    with session_factory.begin() as session:
        for index in range(4):
            job_id = uuid.uuid4()
            run_token = uuid.uuid4()
            job_started = base_time + timedelta(seconds=index * 20)
            completed = job_started + timedelta(seconds=8)
            commit_hash = f"{index + 1:040x}"
            session.add(
                EvolutionJob(
                    id=job_id,
                    status=JobStatus.SUCCEEDED,
                    base_commit_hash="a" * 40,
                    island_id="main",
                    inspiration_commit_hashes=[],
                    result_commit_hash=commit_hash,
                    ingestion_status="skipped",
                    ingestion_attempts=1,
                    started_at=job_started,
                    completed_at=completed,
                    created_at=job_started - timedelta(seconds=1),
                    updated_at=completed,
                )
            )
            record_evolution_event(
                session,
                event_type=JOB_DISPATCHED,
                job_id=job_id,
                island_id="main",
                occurred_at=job_started - timedelta(milliseconds=500),
                ordinal=1,
                payload={
                    "dispatch_kind": "dispatch",
                    "previous_status": "pending",
                    "recovery_count": 0,
                },
                key_parts=("dispatch",),
            )
            if index == 3:
                crashed_run = uuid.uuid4()
                record_evolution_event(
                    session,
                    event_type=JOB_RUN_STARTED,
                    job_id=job_id,
                    run_token=crashed_run,
                    island_id="main",
                    occurred_at=job_started - timedelta(seconds=4),
                    payload={"job_kind": "evolution", "recovery_count": 0},
                    key_parts=("worker_run",),
                )
                start_evolution_stage(
                    session,
                    stage="planning",
                    job_id=job_id,
                    run_token=crashed_run,
                    island_id="main",
                    ordinal=1,
                    occurred_at=job_started - timedelta(seconds=3),
                )
                record_evolution_event(
                    session,
                    event_type=JOB_RECLAIMED,
                    job_id=job_id,
                    run_token=crashed_run,
                    island_id="main",
                    occurred_at=job_started - timedelta(seconds=1),
                    ordinal=1,
                    payload={
                        "reason": "lease_expired",
                        "outcome": "requeued",
                        "recovery_count": 1,
                    },
                    key_parts=("reclaim",),
                )
                record_evolution_event(
                    session,
                    event_type=JOB_DISPATCHED,
                    job_id=job_id,
                    island_id="main",
                    occurred_at=job_started - timedelta(milliseconds=750),
                    ordinal=2,
                    payload={
                        "dispatch_kind": "redispatch",
                        "previous_status": "pending",
                        "recovery_count": 1,
                    },
                    key_parts=("dispatch",),
                )
            record_evolution_event(
                session,
                event_type=JOB_RUN_STARTED,
                job_id=job_id,
                run_token=run_token,
                island_id="main",
                occurred_at=job_started,
                payload={
                    "job_kind": "evolution",
                    "recovery_count": 1 if index == 3 else 0,
                },
                key_parts=("worker_run",),
            )
            planning = start_evolution_stage(
                session,
                stage="planning",
                job_id=job_id,
                run_token=run_token,
                island_id="main",
                ordinal=1,
                occurred_at=job_started + timedelta(seconds=1),
            )
            finish_evolution_stage(
                session,
                handle=planning,
                outcome="succeeded",
                occurred_at=job_started + timedelta(seconds=2),
            )
            coding = start_evolution_stage(
                session,
                stage="coding",
                job_id=job_id,
                run_token=run_token,
                island_id="main",
                ordinal=1,
                occurred_at=job_started + timedelta(seconds=2),
                payload={"rework": False},
            )
            finish_evolution_stage(
                session,
                handle=coding,
                outcome="succeeded",
                payload={"rework": False},
                occurred_at=job_started + timedelta(seconds=3),
            )
            evaluation = start_evolution_stage(
                session,
                stage="evaluation",
                job_id=job_id,
                run_token=run_token,
                island_id="main",
                commit_hash=commit_hash,
                ordinal=1,
                occurred_at=job_started + timedelta(seconds=3),
                payload={"protocol": "one_shot"},
            )
            session.add(
                EvaluationAttempt(
                    job_id=job_id,
                    run_token=run_token,
                    attempt_ordinal=evaluation.ordinal,
                    outcome_kind="passed",
                    protocol="one_shot",
                    started_at=job_started + timedelta(seconds=3),
                    finished_at=job_started + timedelta(seconds=7),
                )
            )
            record_evolution_event(
                session,
                event_type=JOB_SUCCEEDED,
                job_id=job_id,
                run_token=run_token,
                island_id="main",
                commit_hash=commit_hash,
                occurred_at=completed,
                payload={"outcome": "succeeded"},
                key_parts=("terminal",),
            )
            ingestion = start_evolution_stage(
                session,
                stage="ingestion",
                job_id=job_id,
                island_id="main",
                commit_hash=commit_hash,
                ordinal=1,
                occurred_at=completed + timedelta(seconds=1),
            )
            finish_evolution_stage(
                session,
                handle=ingestion,
                outcome="not_admitted",
                payload={
                    "reason": "projection_warmup",
                    "status_code": 0,
                    "inserted": False,
                },
                occurred_at=completed + timedelta(seconds=2),
            )
            record_evolution_event(
                session,
                event_type=ARCHIVE_CANDIDATE_CONSIDERED,
                job_id=job_id,
                island_id="main",
                commit_hash=commit_hash,
                occurred_at=completed + timedelta(seconds=2),
                ordinal=1,
                payload={
                    "outcome": "not_admitted",
                    "reason": "projection_warmup",
                },
                key_parts=("timeline_fixture", index, "archive_considered"),
            )

    with session_factory() as session:
        exported = export_evolution_timeline(session, strict=True)
        usage_count = session.execute(select(func.count(LLMUsageEvent.id))).scalar_one()
        event_count, event_bytes = session.execute(
            text(
                "SELECT count(*), "
                "coalesce(sum(pg_column_size(evolution_events)), 0) "
                "FROM evolution_events"
            )
        ).one()
    fixture_seconds = perf_counter() - fixture_started

    assert exported.issues == ()
    assert usage_count == 0
    assert event_count == 48
    assert int(event_bytes) < 64 * 1024
    assert fixture_seconds < 10.0
    assert "planning.interrupted" in exported.to_jsonl()

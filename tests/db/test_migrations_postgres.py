from __future__ import annotations

import multiprocessing
import os
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from time import monotonic, sleep

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import sessionmaker

from loreley.db.base import INSTANCE_SCHEMA_VERSION
from loreley.db.migrations.runner import (
    MigrationRequiredError,
    ensure_schema_current,
    validate_database_schema,
)
from loreley.core.worker.evaluation_runtime import (
    EvaluationRuntimeCoordinator,
    EvaluationRuntimeError,
)
from loreley.db.base import Base
from loreley.db.models import EvolutionJob, JobStatus
from loreley.naming import resolve_experiment_identity
from tests.support import TestSettings


POSTGRES_TEST_DSN = os.getenv("LORELEY_TEST_DATABASE_URL") or os.getenv("LORELEY_POSTGRES_TEST_DSN")


def _hold_postgres_advisory_lock(
    database_dsn: str,
    advisory_key: int,
    ready: object,
) -> None:
    """Child-process helper used to prove session-lock death recovery."""

    engine = create_engine(database_dsn, future=True)
    try:
        with engine.connect() as connection:
            connection.execute(
                text("SELECT pg_advisory_lock(:key)"),
                {"key": advisory_key},
            )
            ready.send_bytes(b"held")  # type: ignore[attr-defined]
            sleep(60)
    finally:
        engine.dispose()


@pytest.fixture
def postgres_engine() -> Iterator[Engine]:
    if not POSTGRES_TEST_DSN:
        pytest.skip("set LORELEY_TEST_DATABASE_URL or LORELEY_POSTGRES_TEST_DSN to run Postgres migration tests")

    schema_name = f"loreley_migration_test_{uuid.uuid4().hex}"
    admin_engine = create_engine(POSTGRES_TEST_DSN, future=True)
    with admin_engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema_name}"'))

    engine = create_engine(
        POSTGRES_TEST_DSN,
        connect_args={"options": f"-csearch_path={schema_name}"},
        future=True,
    )
    try:
        yield engine
    finally:
        engine.dispose()
        with admin_engine.begin() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE'))
        admin_engine.dispose()


@pytest.fixture
def migration_settings() -> TestSettings:
    return TestSettings(
        EXPERIMENT_ID="migration-demo",
        MAPELITES_EXPERIMENT_ROOT_COMMIT="root000000000000000000000000000000000000000",
        DB_AUTO_MIGRATE=True,
    )


_V5_SCHEMA_STATEMENTS = (
    """
    CREATE TYPE job_status AS ENUM (
        'PENDING',
        'QUEUED',
        'RUNNING',
        'SUCCEEDED',
        'FAILED',
        'CANCELLED'
    )
    """,
    """
    CREATE TABLE instance_metadata (
        id INTEGER PRIMARY KEY,
        schema_version INTEGER NOT NULL,
        experiment_id_raw VARCHAR(128) NOT NULL,
        experiment_uuid UUID NOT NULL,
        root_commit_hash VARCHAR(64) NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        CONSTRAINT ck_instance_metadata_single_row CHECK (id = 1)
    )
    """,
    """
    CREATE TABLE evolution_jobs (
        id UUID PRIMARY KEY,
        status job_status NOT NULL DEFAULT 'PENDING',
        base_commit_hash VARCHAR(64),
        island_id VARCHAR(64),
        inspiration_commit_hashes VARCHAR(64)[] NOT NULL DEFAULT ARRAY[]::VARCHAR(64)[],
        plan_summary TEXT,
        goal VARCHAR(512),
        constraints VARCHAR(200)[] NOT NULL DEFAULT ARRAY[]::VARCHAR(200)[],
        acceptance_criteria VARCHAR(200)[] NOT NULL DEFAULT ARRAY[]::VARCHAR(200)[],
        notes VARCHAR(200)[] NOT NULL DEFAULT ARRAY[]::VARCHAR(200)[],
        tags VARCHAR(64)[] NOT NULL DEFAULT ARRAY[]::VARCHAR(64)[],
        iteration_hint VARCHAR(256),
        sampling_strategy VARCHAR(64),
        sampling_initial_radius INTEGER,
        sampling_radius_used INTEGER,
        sampling_fallback_inspirations INTEGER,
        is_seed_job BOOLEAN NOT NULL DEFAULT FALSE,
        candidate_commit_hash VARCHAR(64),
        candidate_branch_name VARCHAR(255),
        candidate_published_at TIMESTAMP WITH TIME ZONE,
        result_commit_hash VARCHAR(64),
        ingestion_status VARCHAR(32),
        ingestion_attempts INTEGER NOT NULL DEFAULT 0,
        ingestion_delta DOUBLE PRECISION,
        ingestion_status_code INTEGER,
        ingestion_message TEXT,
        ingestion_cell_index INTEGER,
        ingestion_last_attempt_at TIMESTAMP WITH TIME ZONE,
        ingestion_reason TEXT,
        priority INTEGER NOT NULL DEFAULT 0,
        scheduled_at TIMESTAMP WITH TIME ZONE,
        started_at TIMESTAMP WITH TIME ZONE,
        heartbeat_at TIMESTAMP WITH TIME ZONE,
        lease_expires_at TIMESTAMP WITH TIME ZONE,
        run_token UUID,
        worker_id VARCHAR(128),
        recovery_count INTEGER NOT NULL DEFAULT 0,
        completed_at TIMESTAMP WITH TIME ZONE,
        last_error TEXT,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE commit_cards (
        id UUID PRIMARY KEY,
        commit_hash VARCHAR(64) NOT NULL UNIQUE,
        parent_commit_hash VARCHAR(64),
        island_id VARCHAR(64),
        job_id UUID REFERENCES evolution_jobs(id) ON DELETE SET NULL,
        author VARCHAR(128),
        subject VARCHAR(72) NOT NULL,
        change_summary VARCHAR(512) NOT NULL,
        evaluation_summary VARCHAR(512),
        tags VARCHAR(64)[] NOT NULL DEFAULT ARRAY[]::VARCHAR(64)[],
        key_files VARCHAR(256)[] NOT NULL DEFAULT ARRAY[]::VARCHAR(256)[],
        highlights VARCHAR(200)[] NOT NULL DEFAULT ARRAY[]::VARCHAR(200)[],
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE metrics (
        id UUID PRIMARY KEY,
        commit_card_id UUID NOT NULL REFERENCES commit_cards(id) ON DELETE CASCADE,
        name VARCHAR(128) NOT NULL,
        value DOUBLE PRECISION NOT NULL,
        unit VARCHAR(32),
        higher_is_better BOOLEAN NOT NULL DEFAULT TRUE,
        details JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        CONSTRAINT uq_metric_commit_card_name UNIQUE (commit_card_id, name)
    )
    """,
    """
    CREATE TABLE job_artifacts (
        job_id UUID PRIMARY KEY REFERENCES evolution_jobs(id) ON DELETE CASCADE,
        planning_prompt_path VARCHAR(1024),
        planning_raw_output_path VARCHAR(1024),
        planning_plan_json_path VARCHAR(1024),
        coding_prompt_path VARCHAR(1024),
        coding_raw_output_path VARCHAR(1024),
        coding_execution_json_path VARCHAR(1024),
        evaluation_json_path VARCHAR(1024),
        evaluation_logs_path VARCHAR(1024),
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE commit_chunk_summaries (
        start_commit_hash VARCHAR(64) NOT NULL,
        end_commit_hash VARCHAR(64) NOT NULL,
        block_size INTEGER NOT NULL,
        model VARCHAR(255) NOT NULL DEFAULT '',
        step_count INTEGER NOT NULL,
        summary TEXT NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        PRIMARY KEY (start_commit_hash, end_commit_hash, block_size)
    )
    """,
    """
    CREATE TABLE map_elites_archive_cells (
        island_id VARCHAR(64) NOT NULL,
        cell_index INTEGER NOT NULL,
        commit_hash VARCHAR(64) NOT NULL,
        objective DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        measures DOUBLE PRECISION[] NOT NULL DEFAULT ARRAY[]::DOUBLE PRECISION[],
        solution DOUBLE PRECISION[] NOT NULL DEFAULT ARRAY[]::DOUBLE PRECISION[],
        timestamp DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        PRIMARY KEY (island_id, cell_index)
    )
    """,
    """
    CREATE TABLE map_elites_states (
        island_id VARCHAR(64) PRIMARY KEY,
        snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE map_elites_pca_history (
        island_id VARCHAR(64) NOT NULL,
        commit_hash VARCHAR(64) NOT NULL,
        vector DOUBLE PRECISION[] NOT NULL DEFAULT ARRAY[]::DOUBLE PRECISION[],
        embedding_model VARCHAR(255) NOT NULL DEFAULT '',
        last_seen_at DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        PRIMARY KEY (island_id, commit_hash)
    )
    """,
    """
    CREATE TABLE map_elites_file_embedding_cache (
        blob_sha VARCHAR(64) PRIMARY KEY,
        embedding_model VARCHAR(255) NOT NULL,
        dimensions INTEGER NOT NULL,
        vector DOUBLE PRECISION[] NOT NULL DEFAULT ARRAY[]::DOUBLE PRECISION[],
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE map_elites_repo_state_aggregates (
        commit_hash VARCHAR(64) PRIMARY KEY,
        file_count INTEGER NOT NULL DEFAULT 0,
        sum_vector DOUBLE PRECISION[] NOT NULL DEFAULT ARRAY[]::DOUBLE PRECISION[],
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
    )
    """,
)

_V5_INSERT_INSTANCE_METADATA = """
INSERT INTO instance_metadata (
    id,
    schema_version,
    experiment_id_raw,
    experiment_uuid,
    root_commit_hash
)
VALUES (1, 5, :experiment_id_raw, :experiment_uuid, :root_commit_hash)
"""

_V5_INSERT_EVOLUTION_JOBS = """
INSERT INTO evolution_jobs (
    id,
    status,
    base_commit_hash,
    island_id,
    is_seed_job,
    candidate_commit_hash,
    candidate_branch_name,
    candidate_published_at,
    result_commit_hash,
    ingestion_status,
    run_token,
    completed_at
)
VALUES
    (:seed_job, 'SUCCEEDED', :root, 'main', TRUE, NULL, 'candidate/a', now(), 'commit-a', 'succeeded', :run_token_a, now()),
    (:evolution_job, 'SUCCEEDED', 'commit-a', 'main', FALSE, 'commit-b', NULL, NULL, NULL, 'succeeded', :run_token_b, now()),
    (:failed_job, 'FAILED', :root, 'main', FALSE, 'commit-fail', NULL, NULL, NULL, 'failed', :run_token_c, now())
"""

_V5_INSERT_COMMIT_CARDS = """
INSERT INTO commit_cards (
    id,
    commit_hash,
    parent_commit_hash,
    island_id,
    job_id,
    subject,
    change_summary
)
VALUES
    (:root_card, :root, NULL, 'main', NULL, 'root', 'root baseline'),
    (:commit_a_card, 'commit-a', :root, 'main', :seed_job, 'commit a', 'first historical success')
"""

_V5_INSERT_METRIC = """
INSERT INTO metrics (id, commit_card_id, name, value)
VALUES (:metric, :commit_a_card, 'fitness', 1.0)
"""

_V5_INSERT_ARCHIVE_CELL = """
INSERT INTO map_elites_archive_cells (
    island_id,
    cell_index,
    commit_hash,
    objective
)
VALUES ('main', 1, 'commit-a', 1.0)
"""


def _create_v5_fixture(engine: Engine, settings: TestSettings) -> dict[str, uuid.UUID]:
    identity = resolve_experiment_identity(settings.experiment_id)
    root_commit = str(settings.mapelites_experiment_root_commit)
    ids = {
        "root_card": uuid.uuid4(),
        "commit_a_card": uuid.uuid4(),
        "metric": uuid.uuid4(),
        "seed_job": uuid.uuid4(),
        "evolution_job": uuid.uuid4(),
        "failed_job": uuid.uuid4(),
    }
    with engine.begin() as conn:
        _execute_v5_schema(conn)
        _insert_v5_fixture_rows(conn, ids=ids, identity=identity, root_commit=root_commit)
    return ids


def _execute_v5_schema(conn: Connection) -> None:
    for statement in _V5_SCHEMA_STATEMENTS:
        conn.execute(text(statement))


def _insert_v5_fixture_rows(
    conn: Connection,
    *,
    ids: dict[str, uuid.UUID],
    identity,
    root_commit: str,
) -> None:
    conn.execute(
        text(_V5_INSERT_INSTANCE_METADATA),
        {
            "experiment_id_raw": identity.raw,
            "experiment_uuid": identity.uuid,
            "root_commit_hash": root_commit,
        },
    )
    conn.execute(
        text(_V5_INSERT_EVOLUTION_JOBS),
        {
            "seed_job": ids["seed_job"],
            "evolution_job": ids["evolution_job"],
            "failed_job": ids["failed_job"],
            "root": root_commit,
            "run_token_a": uuid.uuid4(),
            "run_token_b": uuid.uuid4(),
            "run_token_c": uuid.uuid4(),
        },
    )
    conn.execute(
        text(_V5_INSERT_COMMIT_CARDS),
        {
            "root_card": ids["root_card"],
            "commit_a_card": ids["commit_a_card"],
            "root": root_commit,
            "seed_job": ids["seed_job"],
        },
    )
    conn.execute(
        text(_V5_INSERT_METRIC),
        {"metric": ids["metric"], "commit_a_card": ids["commit_a_card"]},
    )
    conn.execute(text(_V5_INSERT_ARCHIVE_CELL))


def test_fresh_database_path_seeds_current_schema_version(
    postgres_engine: Engine,
    migration_settings: TestSettings,
) -> None:
    result = ensure_schema_current(
        engine=postgres_engine,
        settings=migration_settings,
        target_version=INSTANCE_SCHEMA_VERSION,
        auto_migrate=True,
    )

    assert result.fresh_database is True
    assert result.to_version == INSTANCE_SCHEMA_VERSION
    with postgres_engine.connect() as conn:
        version = conn.execute(text("SELECT schema_version FROM instance_metadata WHERE id = 1")).scalar_one()
        audit = conn.execute(
            text("SELECT name FROM loreley_schema_migrations WHERE version = :version"),
            {"version": INSTANCE_SCHEMA_VERSION},
        ).scalar_one()
    assert version == INSTANCE_SCHEMA_VERSION
    assert audit == "fresh_create_all"


def test_v5_fixture_migrates_to_current_preserves_rows_and_backfills_candidates(
    postgres_engine: Engine,
    migration_settings: TestSettings,
) -> None:
    ids = _create_v5_fixture(postgres_engine, migration_settings)

    result = ensure_schema_current(
        engine=postgres_engine,
        settings=migration_settings,
        target_version=INSTANCE_SCHEMA_VERSION,
        auto_migrate=True,
    )

    assert result.from_version == 5
    assert result.applied_versions == (
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
    )
    validate_database_schema(
        engine=postgres_engine,
        settings=migration_settings,
        target_version=INSTANCE_SCHEMA_VERSION,
    )
    with postgres_engine.connect() as conn:
        assert (
            conn.execute(text("SELECT schema_version FROM instance_metadata WHERE id = 1")).scalar_one()
            == INSTANCE_SCHEMA_VERSION
        )
        assert conn.execute(text("SELECT count(*) FROM commit_cards")).scalar_one() == 2
        assert conn.execute(text("SELECT count(*) FROM metrics")).scalar_one() == 1
        assert conn.execute(text("SELECT count(*) FROM evolution_jobs")).scalar_one() == 3
        # v15 intentionally discards the old scalar-per-cell archive. Durable
        # successful jobs are marked for reingestion into Pareto fronts.
        assert conn.execute(text("SELECT count(*) FROM map_elites_archive_cells")).scalar_one() == 0
        job_kinds = {
            str(row["id"]): row["job_kind"]
            for row in conn.execute(text("SELECT id, job_kind FROM evolution_jobs")).mappings()
        }
        migrated_jobs = {
            str(row["id"]): row
            for row in conn.execute(
                text(
                    """
                    SELECT id, result_commit_hash, ingestion_status
                    FROM evolution_jobs
                    """
                )
            ).mappings()
        }
        candidates = {
            row["commit_hash"]: row
            for row in conn.execute(
                text(
                    """
                    SELECT
                        commit_hash,
                        git_parent_commit_hash,
                        nearest_viable_ancestor_hash,
                        produced_by_job_id,
                        evaluation_status,
                        archive_status,
                        repair_state,
                        repair_source_candidate_id,
                        commit_card_id,
                        campaign_program_hash
                    FROM candidate_commits
                    ORDER BY commit_hash
                    """
                )
            ).mappings()
        }
        audit_versions = [
            row[0]
            for row in conn.execute(
                text("SELECT version FROM loreley_schema_migrations ORDER BY version")
            )
        ]
        change_summary_limit = conn.execute(
            text(
                """
                    SELECT character_maximum_length
                    FROM information_schema.columns
                    WHERE table_schema = current_schema()
                      AND table_name = 'commit_cards'
                      AND column_name = 'change_summary'
                """
            )
        ).scalar_one()

    assert job_kinds[str(ids["seed_job"])] == "seed"
    assert job_kinds[str(ids["evolution_job"])] == "evolution"
    assert job_kinds[str(ids["failed_job"])] == "evolution"
    assert migrated_jobs[str(ids["seed_job"])]["result_commit_hash"] == "commit-a"
    assert migrated_jobs[str(ids["seed_job"])]["ingestion_status"] is None
    assert migrated_jobs[str(ids["evolution_job"])]["result_commit_hash"] == "commit-b"
    assert migrated_jobs[str(ids["evolution_job"])]["ingestion_status"] is None
    assert set(candidates) == {"commit-a", "commit-b"}
    assert candidates["commit-a"]["evaluation_status"] == "passed"
    assert candidates["commit-a"]["archive_status"] == "not_considered"
    assert candidates["commit-a"]["commit_card_id"] is not None
    assert candidates["commit-b"]["archive_status"] == "not_considered"
    assert candidates["commit-b"]["git_parent_commit_hash"] == "commit-a"
    assert candidates["commit-b"]["nearest_viable_ancestor_hash"] == "commit-a"
    assert candidates["commit-b"]["repair_state"] == "audit_only"
    assert candidates["commit-b"]["repair_source_candidate_id"] is None
    assert candidates["commit-b"]["campaign_program_hash"] is None
    assert audit_versions == [
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
    ]
    assert change_summary_limit == 800


def test_migration_is_idempotent_after_v5_upgrade(
    postgres_engine: Engine,
    migration_settings: TestSettings,
) -> None:
    _create_v5_fixture(postgres_engine, migration_settings)
    ensure_schema_current(
        engine=postgres_engine,
        settings=migration_settings,
        target_version=INSTANCE_SCHEMA_VERSION,
        auto_migrate=True,
    )
    with postgres_engine.connect() as conn:
        first_candidate_count = conn.execute(text("SELECT count(*) FROM candidate_commits")).scalar_one()

    result = ensure_schema_current(
        engine=postgres_engine,
        settings=migration_settings,
        target_version=INSTANCE_SCHEMA_VERSION,
        auto_migrate=True,
    )

    with postgres_engine.connect() as conn:
        second_candidate_count = conn.execute(text("SELECT count(*) FROM candidate_commits")).scalar_one()
    assert result.applied_versions == ()
    assert second_candidate_count == first_candidate_count


def test_auto_migrate_disabled_fails_with_migrate_hint(
    postgres_engine: Engine,
    migration_settings: TestSettings,
) -> None:
    _create_v5_fixture(postgres_engine, migration_settings)

    with pytest.raises(MigrationRequiredError, match="uv run loreley db migrate"):
        ensure_schema_current(
            engine=postgres_engine,
            settings=migration_settings,
            target_version=INSTANCE_SCHEMA_VERSION,
            auto_migrate=False,
        )

    with postgres_engine.connect() as conn:
        version = conn.execute(text("SELECT schema_version FROM instance_metadata WHERE id = 1")).scalar_one()
    assert version == 5


def test_evaluator_slots_are_global_and_release_for_waiters(
    postgres_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two worker coordinators share E=1 through a PostgreSQL advisory slot."""

    import loreley.core.worker.evaluation_runtime as runtime

    Base.metadata.create_all(postgres_engine)
    with postgres_engine.connect() as conn:
        schema = str(conn.execute(text("SELECT current_schema()")).scalar_one())
    scoped_dsn = postgres_engine.url.update_query_dict(
        {"options": f"-csearch_path={schema}"}
    ).render_as_string(hide_password=False)
    session_factory = sessionmaker(bind=postgres_engine, expire_on_commit=False, future=True)

    @contextmanager
    def scoped_session():
        with session_factory.begin() as session:
            yield session

    monkeypatch.setattr(runtime, "session_scope", scoped_session)
    runtime._lock_engine.cache_clear()  # noqa: SLF001 - isolated schema contract
    first_job, second_job = uuid.uuid4(), uuid.uuid4()
    first_token, second_token = uuid.uuid4(), uuid.uuid4()
    with session_factory.begin() as session:
        session.add_all(
            [
                EvolutionJob(id=first_job, status=JobStatus.RUNNING, run_token=first_token),
                EvolutionJob(id=second_job, status=JobStatus.RUNNING, run_token=second_token),
            ]
        )
    settings = TestSettings(
        DATABASE_URL=scoped_dsn,
        EXPERIMENT_ID="slot-test",
        WORKER_EVALUATOR_MAX_CONCURRENCY=1,
    )
    first = EvaluationRuntimeCoordinator(settings)
    second = EvaluationRuntimeCoordinator(settings)
    contract = first.ensure_contract(
        evaluator_name="demo",
        evaluator_version="1",
        campaign_program_hash="a" * 64,
        limit_scope="measurement",
    )
    held = first.acquire_evaluator_slot(
        contract_key=contract,
        job_id=first_job,
        run_token=first_token,
        deadline=monotonic() + 5,
    )
    assert held is not None
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiting = pool.submit(
            second.acquire_evaluator_slot,
            contract_key=contract,
            job_id=second_job,
            run_token=second_token,
            deadline=monotonic() + 5,
        )
        sleep(0.1)
        assert not waiting.done()
        held.release("test_release")
        acquired = waiting.result(timeout=5)
    assert acquired is not None
    assert acquired.slot_index == 0
    assert acquired.wait_seconds >= 0.05
    acquired.release("test_complete")
    mismatched_e = EvaluationRuntimeCoordinator(
        TestSettings(
            DATABASE_URL=scoped_dsn,
            EXPERIMENT_ID="slot-test",
            WORKER_EVALUATOR_MAX_CONCURRENCY=2,
        )
    )
    with pytest.raises(EvaluationRuntimeError, match="MAX_CONCURRENCY disagrees"):
        mismatched_e.ensure_contract(
            evaluator_name="demo",
            evaluator_version="1",
            campaign_program_hash="a" * 64,
            limit_scope="measurement",
        )
    with pytest.raises(EvaluationRuntimeError, match="limit scope disagrees"):
        first.ensure_contract(
            evaluator_name="demo",
            evaluator_version="1",
            campaign_program_hash="a" * 64,
            limit_scope="whole",
        )
    runtime._lock_engine.cache_clear()  # noqa: SLF001


def test_postgres_evaluator_slot_is_released_when_holder_process_dies(
    postgres_engine: Engine,
) -> None:
    """PostgreSQL releases the framework's session lock after an abrupt worker death."""

    database_dsn = postgres_engine.url.render_as_string(hide_password=False)
    advisory_key = int.from_bytes(uuid.uuid4().bytes[:8], byteorder="big", signed=True)
    context = multiprocessing.get_context("spawn")
    child_watch, parent_watch = context.Pipe(duplex=False)
    process = context.Process(
        target=_hold_postgres_advisory_lock,
        args=(database_dsn, advisory_key, parent_watch),
    )
    process.start()
    parent_watch.close()
    try:
        assert child_watch.poll(10), "child did not acquire the advisory lock"
        assert child_watch.recv_bytes() == b"held"
        process.kill()
        process.join(timeout=10)
        assert not process.is_alive()

        with postgres_engine.connect() as connection:
            release_deadline = monotonic() + 5
            acquired = False
            while not acquired and monotonic() < release_deadline:
                acquired = bool(
                    connection.execute(
                        text("SELECT pg_try_advisory_lock(:key)"),
                        {"key": advisory_key},
                    ).scalar_one()
                )
                if not acquired:
                    sleep(0.05)
            assert acquired, "PostgreSQL did not release the dead session's advisory lock"
            connection.execute(
                text("SELECT pg_advisory_unlock(:key)"),
                {"key": advisory_key},
            )
    finally:
        child_watch.close()
        if process.is_alive():
            process.kill()
            process.join(timeout=10)

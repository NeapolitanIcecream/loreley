from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from loreley.db.base import INSTANCE_SCHEMA_VERSION
from loreley.db.migrations.runner import (
    MigrationRequiredError,
    ensure_schema_current,
    validate_database_schema,
)
from loreley.naming import resolve_experiment_identity
from tests.support import TestSettings


POSTGRES_TEST_DSN = os.getenv("LORELEY_TEST_DATABASE_URL") or os.getenv("LORELEY_POSTGRES_TEST_DSN")


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
        conn.execute(
            text(
                """
                CREATE TYPE job_status AS ENUM (
                    'PENDING',
                    'QUEUED',
                    'RUNNING',
                    'SUCCEEDED',
                    'FAILED',
                    'CANCELLED'
                )
                """
            )
        )
        conn.execute(
            text(
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
                """
            )
        )
        conn.execute(
            text(
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
                """
            )
        )
        conn.execute(
            text(
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
                """
            )
        )
        conn.execute(
            text(
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
                """
            )
        )
        conn.execute(
            text(
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
                """
            )
        )
        conn.execute(
            text(
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
                """
            )
        )
        conn.execute(
            text(
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
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE map_elites_states (
                    island_id VARCHAR(64) PRIMARY KEY,
                    snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
                )
                """
            )
        )
        conn.execute(
            text(
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
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE map_elites_file_embedding_cache (
                    blob_sha VARCHAR(64) PRIMARY KEY,
                    embedding_model VARCHAR(255) NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector DOUBLE PRECISION[] NOT NULL DEFAULT ARRAY[]::DOUBLE PRECISION[],
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE map_elites_repo_state_aggregates (
                    commit_hash VARCHAR(64) PRIMARY KEY,
                    file_count INTEGER NOT NULL DEFAULT 0,
                    sum_vector DOUBLE PRECISION[] NOT NULL DEFAULT ARRAY[]::DOUBLE PRECISION[],
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
                )
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO instance_metadata (
                    id,
                    schema_version,
                    experiment_id_raw,
                    experiment_uuid,
                    root_commit_hash
                )
                VALUES (1, 5, :experiment_id_raw, :experiment_uuid, :root_commit_hash)
                """
            ),
            {
                "experiment_id_raw": identity.raw,
                "experiment_uuid": identity.uuid,
                "root_commit_hash": root_commit,
            },
        )
        conn.execute(
            text(
                """
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
            ),
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
            text(
                """
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
            ),
            {
                "root_card": ids["root_card"],
                "commit_a_card": ids["commit_a_card"],
                "root": root_commit,
                "seed_job": ids["seed_job"],
            },
        )
        conn.execute(
            text(
                """
                INSERT INTO metrics (id, commit_card_id, name, value)
                VALUES (:metric, :commit_a_card, 'fitness', 1.0)
                """
            ),
            {"metric": ids["metric"], "commit_a_card": ids["commit_a_card"]},
        )
        conn.execute(
            text(
                """
                INSERT INTO map_elites_archive_cells (
                    island_id,
                    cell_index,
                    commit_hash,
                    objective
                )
                VALUES ('main', 1, 'commit-a', 1.0)
                """
            )
        )
    return ids


def test_fresh_database_path_seeds_schema_version_12(
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
    assert result.to_version == 12
    with postgres_engine.connect() as conn:
        version = conn.execute(text("SELECT schema_version FROM instance_metadata WHERE id = 1")).scalar_one()
        audit = conn.execute(
            text("SELECT name FROM loreley_schema_migrations WHERE version = 12")
        ).scalar_one()
    assert version == 12
    assert audit == "fresh_create_all"


def test_v5_fixture_migrates_to_v12_preserves_rows_and_backfills_candidates(
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
    assert result.applied_versions == (6, 7, 8, 9, 10, 11, 12)
    validate_database_schema(
        engine=postgres_engine,
        settings=migration_settings,
        target_version=INSTANCE_SCHEMA_VERSION,
    )
    with postgres_engine.connect() as conn:
        assert conn.execute(text("SELECT schema_version FROM instance_metadata WHERE id = 1")).scalar_one() == 12
        assert conn.execute(text("SELECT count(*) FROM commit_cards")).scalar_one() == 2
        assert conn.execute(text("SELECT count(*) FROM metrics")).scalar_one() == 1
        assert conn.execute(text("SELECT count(*) FROM evolution_jobs")).scalar_one() == 3
        assert conn.execute(text("SELECT count(*) FROM map_elites_archive_cells")).scalar_one() == 1
        job_kinds = {
            str(row["id"]): row["job_kind"]
            for row in conn.execute(text("SELECT id, job_kind FROM evolution_jobs")).mappings()
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

    assert job_kinds[str(ids["seed_job"])] == "seed"
    assert job_kinds[str(ids["evolution_job"])] == "evolution"
    assert job_kinds[str(ids["failed_job"])] == "evolution"
    assert set(candidates) == {"commit-a", "commit-b"}
    assert candidates["commit-a"]["evaluation_status"] == "passed"
    assert candidates["commit-a"]["archive_status"] == "member"
    assert candidates["commit-a"]["commit_card_id"] is not None
    assert candidates["commit-b"]["archive_status"] == "not_considered"
    assert candidates["commit-b"]["git_parent_commit_hash"] == "commit-a"
    assert candidates["commit-b"]["nearest_viable_ancestor_hash"] == "commit-a"
    assert candidates["commit-b"]["repair_state"] == "audit_only"
    assert candidates["commit-b"]["repair_source_candidate_id"] is None
    assert candidates["commit-b"]["campaign_program_hash"] is None
    assert audit_versions == [6, 7, 8, 9, 10, 11, 12]


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

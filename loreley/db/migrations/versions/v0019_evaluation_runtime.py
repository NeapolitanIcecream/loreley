from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from loreley.config import Settings
from loreley.db.migrations.registry import SchemaMigration


_TABLE_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS evaluation_measurements (
        id UUID PRIMARY KEY,
        cache_key VARCHAR(64) NOT NULL,
        candidate_identity VARCHAR(512) NOT NULL,
        evaluation_identity_key VARCHAR(64) NOT NULL,
        evaluator_name VARCHAR(128) NOT NULL,
        evaluator_version VARCHAR(128) NOT NULL,
        campaign_program_hash VARCHAR(64) NOT NULL,
        measurement_contract_fingerprint VARCHAR(512) NOT NULL,
        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
        payload_sha256 VARCHAR(64) NOT NULL,
        evidence_manifest JSONB NOT NULL DEFAULT '[]'::jsonb,
        source_job_id UUID REFERENCES evolution_jobs(id) ON DELETE SET NULL,
        source_candidate_commit_hash VARCHAR(64),
        source_evaluation_attempt_id UUID,
        accepted_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        CONSTRAINT uq_evaluation_measurements_cache_key UNIQUE (cache_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evaluation_concurrency_contracts (
        contract_key VARCHAR(64) PRIMARY KEY,
        experiment_id VARCHAR(128) NOT NULL,
        evaluator_name VARCHAR(128) NOT NULL,
        evaluator_version VARCHAR(128) NOT NULL,
        campaign_program_hash VARCHAR(64) NOT NULL,
        max_concurrency INTEGER,
        limit_scope VARCHAR(32) NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evaluation_resource_leases (
        id UUID PRIMARY KEY,
        resource_kind VARCHAR(32) NOT NULL,
        resource_key VARCHAR(128) NOT NULL,
        contract_key VARCHAR(64) REFERENCES evaluation_concurrency_contracts(contract_key)
            ON DELETE SET NULL,
        slot_index INTEGER,
        job_id UUID REFERENCES evolution_jobs(id) ON DELETE SET NULL,
        run_token UUID,
        worker_id VARCHAR(128),
        status VARCHAR(32) NOT NULL DEFAULT 'waiting',
        requested_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        acquired_at TIMESTAMP WITH TIME ZONE,
        released_at TIMESTAMP WITH TIME ZONE,
        wait_seconds DOUBLE PRECISION,
        release_reason VARCHAR(64),
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
    )
    """,
)

_COLUMN_STATEMENTS = (
    """
    ALTER TABLE evolution_jobs
        ADD COLUMN IF NOT EXISTS failure_stage VARCHAR(32),
        ADD COLUMN IF NOT EXISTS failure_kind VARCHAR(64)
    """,
    """
    ALTER TABLE evaluation_attempts
        ADD COLUMN IF NOT EXISTS protocol VARCHAR(32) NOT NULL DEFAULT 'one_shot',
        ADD COLUMN IF NOT EXISTS measurement_cache_key VARCHAR(64),
        ADD COLUMN IF NOT EXISTS measurement_contract_fingerprint VARCHAR(512),
        ADD COLUMN IF NOT EXISTS measurement_id UUID,
        ADD COLUMN IF NOT EXISTS measurement_reused BOOLEAN NOT NULL DEFAULT FALSE,
        ADD COLUMN IF NOT EXISTS measurement_executed BOOLEAN NOT NULL DEFAULT FALSE,
        ADD COLUMN IF NOT EXISTS reuse_kind VARCHAR(32) NOT NULL DEFAULT 'none',
        ADD COLUMN IF NOT EXISTS reused_from_attempt_id UUID,
        ADD COLUMN IF NOT EXISTS evaluator_slot INTEGER,
        ADD COLUMN IF NOT EXISTS evaluator_slot_scope VARCHAR(32),
        ADD COLUMN IF NOT EXISTS evaluator_slot_wait_seconds DOUBLE PRECISION,
        ADD COLUMN IF NOT EXISTS evaluator_slot_acquired_at TIMESTAMP WITH TIME ZONE,
        ADD COLUMN IF NOT EXISTS evaluator_slot_released_at TIMESTAMP WITH TIME ZONE,
        ADD COLUMN IF NOT EXISTS evaluator_slot_lease_id UUID,
        ADD COLUMN IF NOT EXISTS evaluator_slot_release_reason VARCHAR(64)
    """,
    """
    ALTER TABLE evaluation_artifacts
        ADD COLUMN IF NOT EXISTS evaluation_attempt_id UUID
    """,
)

_CONSTRAINT_STATEMENTS = (
    """DO $$ BEGIN
        ALTER TABLE evaluation_attempts
            ADD CONSTRAINT fk_evaluation_attempts_measurement_id
            FOREIGN KEY (measurement_id)
            REFERENCES evaluation_measurements(id)
            ON DELETE SET NULL;
    EXCEPTION WHEN duplicate_object THEN NULL; END $$""",
    """DO $$ BEGIN
        ALTER TABLE evaluation_attempts
            ADD CONSTRAINT fk_evaluation_attempts_reused_from_attempt_id
            FOREIGN KEY (reused_from_attempt_id)
            REFERENCES evaluation_attempts(id)
            ON DELETE SET NULL;
    EXCEPTION WHEN duplicate_object THEN NULL; END $$""",
    """DO $$ BEGIN
        ALTER TABLE evaluation_measurements
            ADD CONSTRAINT fk_evaluation_measurements_source_attempt_id
            FOREIGN KEY (source_evaluation_attempt_id)
            REFERENCES evaluation_attempts(id)
            ON DELETE SET NULL;
    EXCEPTION WHEN duplicate_object THEN NULL; END $$""",
    """DO $$ BEGIN
        ALTER TABLE evaluation_attempts
            ADD CONSTRAINT fk_evaluation_attempts_slot_lease_id
            FOREIGN KEY (evaluator_slot_lease_id)
            REFERENCES evaluation_resource_leases(id)
            ON DELETE SET NULL;
    EXCEPTION WHEN duplicate_object THEN NULL; END $$""",
    """DO $$ BEGIN
        ALTER TABLE evaluation_artifacts
            ADD CONSTRAINT fk_evaluation_artifacts_attempt_id
            FOREIGN KEY (evaluation_attempt_id)
            REFERENCES evaluation_attempts(id)
            ON DELETE SET NULL;
    EXCEPTION WHEN duplicate_object THEN NULL; END $$""",
)

_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS ix_evaluation_attempts_measurement_id ON evaluation_attempts (measurement_id)",
    "CREATE INDEX IF NOT EXISTS ix_evaluation_attempts_reused_from ON evaluation_attempts (reused_from_attempt_id)",
    "CREATE INDEX IF NOT EXISTS ix_evaluation_artifacts_attempt_id ON evaluation_artifacts (evaluation_attempt_id)",
    "CREATE INDEX IF NOT EXISTS ix_evaluation_measurements_identity_key ON evaluation_measurements (evaluation_identity_key)",
    "CREATE INDEX IF NOT EXISTS ix_evaluation_measurements_source_attempt ON evaluation_measurements (source_evaluation_attempt_id)",
    "CREATE INDEX IF NOT EXISTS ix_evaluation_resource_leases_status_requested ON evaluation_resource_leases (status, requested_at)",
    "CREATE INDEX IF NOT EXISTS ix_evaluation_resource_leases_job_id ON evaluation_resource_leases (job_id)",
    "CREATE INDEX IF NOT EXISTS ix_evaluation_resource_leases_resource ON evaluation_resource_leases (resource_kind, resource_key)",
)


def _execute_statements(conn: Connection, statements: tuple[str, ...]) -> None:
    for statement in statements:
        conn.execute(text(statement))


def upgrade(conn: Connection, settings: Settings) -> None:
    """Add phased-measurement provenance and global evaluator controls."""

    del settings
    _execute_statements(conn, _TABLE_STATEMENTS)
    _execute_statements(conn, _COLUMN_STATEMENTS)
    _execute_statements(conn, _CONSTRAINT_STATEMENTS)
    _execute_statements(conn, _INDEX_STATEMENTS)


MIGRATION = SchemaMigration(
    from_version=18,
    to_version=19,
    name="v0019_evaluation_runtime",
    upgrade=upgrade,
)

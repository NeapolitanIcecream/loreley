from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from loreley.config import Settings
from loreley.db.migrations.registry import SchemaMigration


def upgrade(conn: Connection, settings: Settings) -> None:
    """Add first-class supplied-candidate and manual-seed job fields."""

    del settings
    conn.execute(text("ALTER TYPE job_status ADD VALUE IF NOT EXISTS 'STAGED'"))
    conn.execute(
        text(
            """
            ALTER TABLE evolution_jobs
                ADD COLUMN IF NOT EXISTS execution_mode VARCHAR(32) NOT NULL DEFAULT 'agent',
                ADD COLUMN IF NOT EXISTS input_candidate_commit_hash VARCHAR(64),
                ADD COLUMN IF NOT EXISTS input_candidate_summary TEXT,
                ADD COLUMN IF NOT EXISTS external_submission_key VARCHAR(64) NOT NULL DEFAULT '',
                ADD COLUMN IF NOT EXISTS input_provenance JSONB NOT NULL DEFAULT '{}'::jsonb,
                ADD COLUMN IF NOT EXISTS archive_ingestion_enabled BOOLEAN NOT NULL DEFAULT TRUE
            """
        )
    )
    conn.execute(
        text(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'ck_evolution_jobs_execution_input'
                ) THEN
                    ALTER TABLE evolution_jobs
                    ADD CONSTRAINT ck_evolution_jobs_execution_input CHECK (
                        (execution_mode = 'agent' AND input_candidate_commit_hash IS NULL)
                        OR
                        (execution_mode = 'evaluate_existing'
                         AND input_candidate_commit_hash IS NOT NULL)
                    );
                END IF;
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'ck_evolution_jobs_manual_seed_contract'
                ) THEN
                    ALTER TABLE evolution_jobs
                    ADD CONSTRAINT ck_evolution_jobs_manual_seed_contract CHECK (
                        job_kind <> 'manual_seed'
                        OR
                        (execution_mode = 'evaluate_existing'
                         AND is_seed_job = TRUE
                         AND archive_ingestion_enabled = TRUE)
                    );
                END IF;
            END
            $$
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_evolution_jobs_manual_seed_commit
                ON evolution_jobs (input_candidate_commit_hash)
                WHERE job_kind = 'manual_seed' AND input_candidate_commit_hash IS NOT NULL
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_evolution_jobs_external_submission_key
                ON evolution_jobs (external_submission_key)
                WHERE external_submission_key <> ''
            """
        )
    )


MIGRATION = SchemaMigration(
    from_version=19,
    to_version=20,
    name="v0020_supplied_candidates",
    upgrade=upgrade,
)

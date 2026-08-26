from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from loreley.config import Settings
from loreley.db.migrations.registry import SchemaMigration


def upgrade(conn: Connection, settings: Settings) -> None:
    """Persist seed portfolios and direction/admission provenance."""

    del settings
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS seed_portfolios (
                id UUID PRIMARY KEY,
                request_fingerprint VARCHAR(64) NOT NULL,
                portfolio_hash VARCHAR(64),
                schema_version INTEGER NOT NULL DEFAULT 1,
                status VARCHAR(32) NOT NULL,
                root_commit_hash VARCHAR(64) NOT NULL,
                campaign_program_hash VARCHAR(64),
                baseline_key_hash VARCHAR(64) NOT NULL,
                objective_contract_fingerprint VARCHAR(64) NOT NULL,
                input_evidence_fingerprints JSONB NOT NULL DEFAULT '{}'::jsonb,
                model_backend VARCHAR(128) NOT NULL,
                model_provider VARCHAR(128) NOT NULL,
                model_name VARCHAR(255) NOT NULL,
                reasoning_effort VARCHAR(32) NOT NULL,
                direction_count INTEGER NOT NULL,
                payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                planner_prompt_sha256 VARCHAR(64),
                planner_output_sha256 VARCHAR(64),
                planner_attempts INTEGER NOT NULL DEFAULT 0,
                planner_duration_seconds DOUBLE PRECISION,
                error_summary TEXT,
                planning_started_at TIMESTAMP WITH TIME ZONE,
                completed_at TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                CONSTRAINT uq_seed_portfolios_request_fingerprint
                    UNIQUE (request_fingerprint),
                CONSTRAINT uq_seed_portfolios_portfolio_hash
                    UNIQUE (portfolio_hash)
            )
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS seed_directions (
                id UUID PRIMARY KEY,
                portfolio_id UUID NOT NULL REFERENCES seed_portfolios(id)
                    ON DELETE CASCADE,
                direction_id VARCHAR(64) NOT NULL,
                ordinal INTEGER NOT NULL,
                content_hash VARCHAR(64) NOT NULL,
                title VARCHAR(120) NOT NULL,
                causal_mechanism TEXT NOT NULL,
                admission_intent VARCHAR(64) NOT NULL,
                payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                CONSTRAINT uq_seed_directions_portfolio_direction
                    UNIQUE (portfolio_id, direction_id),
                CONSTRAINT uq_seed_directions_portfolio_ordinal
                    UNIQUE (portfolio_id, ordinal),
                CONSTRAINT uq_seed_directions_portfolio_content
                    UNIQUE (portfolio_id, content_hash)
            )
            """
        )
    )
    conn.execute(
        text(
            """
            ALTER TABLE evolution_jobs
                ADD COLUMN IF NOT EXISTS seed_portfolio_hash VARCHAR(64),
                ADD COLUMN IF NOT EXISTS seed_direction_id VARCHAR(64),
                ADD COLUMN IF NOT EXISTS seed_direction_payload JSONB
                    NOT NULL DEFAULT '{}'::jsonb,
                ADD COLUMN IF NOT EXISTS seed_admission_lane VARCHAR(64),
                ADD COLUMN IF NOT EXISTS seed_admission_reason TEXT
            """
        )
    )
    conn.execute(
        text(
            """
            ALTER TABLE commit_cards
                ADD COLUMN IF NOT EXISTS seed_portfolio_hash VARCHAR(64),
                ADD COLUMN IF NOT EXISTS seed_direction_id VARCHAR(64),
                ADD COLUMN IF NOT EXISTS seed_admission_lane VARCHAR(64),
                ADD COLUMN IF NOT EXISTS seed_admission_reason TEXT
            """
        )
    )
    conn.execute(
        text(
            """
            ALTER TABLE candidate_commits
                ADD COLUMN IF NOT EXISTS seed_portfolio_hash VARCHAR(64),
                ADD COLUMN IF NOT EXISTS seed_direction_id VARCHAR(64),
                ADD COLUMN IF NOT EXISTS seed_admission_lane VARCHAR(64),
                ADD COLUMN IF NOT EXISTS seed_admission_reason TEXT
            """
        )
    )
    conn.execute(
        text(
            """
            ALTER TABLE evaluation_attempts
                ADD COLUMN IF NOT EXISTS seed_portfolio_hash VARCHAR(64),
                ADD COLUMN IF NOT EXISTS seed_direction_id VARCHAR(64)
            """
        )
    )
    for statement in (
        (
            "CREATE INDEX IF NOT EXISTS ix_seed_portfolios_status "
            "ON seed_portfolios (status)"
        ),
        (
            "CREATE INDEX IF NOT EXISTS ix_seed_portfolios_root_commit "
            "ON seed_portfolios (root_commit_hash)"
        ),
        (
            "CREATE INDEX IF NOT EXISTS ix_seed_directions_portfolio "
            "ON seed_directions (portfolio_id, ordinal)"
        ),
        (
            "CREATE INDEX IF NOT EXISTS ix_evolution_jobs_seed_direction "
            "ON evolution_jobs (seed_portfolio_hash, seed_direction_id)"
        ),
        (
            "CREATE INDEX IF NOT EXISTS ix_commit_cards_seed_direction "
            "ON commit_cards (seed_portfolio_hash, seed_direction_id)"
        ),
        (
            "CREATE INDEX IF NOT EXISTS ix_candidate_commits_seed_direction "
            "ON candidate_commits (seed_portfolio_hash, seed_direction_id)"
        ),
        (
            "CREATE INDEX IF NOT EXISTS ix_evaluation_attempts_seed_direction "
            "ON evaluation_attempts (seed_portfolio_hash, seed_direction_id)"
        ),
    ):
        conn.execute(text(statement))


MIGRATION = SchemaMigration(
    from_version=21,
    to_version=22,
    name="v0022_seed_portfolios",
    upgrade=upgrade,
)

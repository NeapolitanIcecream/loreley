from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from loreley.config import Settings
from loreley.db.migrations.registry import SchemaMigration


def upgrade(conn: Connection, _settings: Settings) -> None:
    """Create agent-visible evaluation artifact records."""

    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS evaluation_artifacts (
                id UUID PRIMARY KEY,
                job_id UUID NOT NULL REFERENCES evolution_jobs(id) ON DELETE CASCADE,
                commit_card_id UUID NULL REFERENCES commit_cards(id) ON DELETE SET NULL,
                commit_hash VARCHAR(64) NOT NULL,
                key VARCHAR(128) NOT NULL,
                kind VARCHAR(64) NOT NULL,
                mime_type VARCHAR(128) NOT NULL,
                label VARCHAR(128),
                summary VARCHAR(1024),
                visibility VARCHAR(32) NOT NULL,
                agent_projection VARCHAR(32) NOT NULL,
                storage_path VARCHAR(1024),
                size_bytes BIGINT,
                sha256 VARCHAR(64),
                diagnostics JSONB NOT NULL DEFAULT '[]'::jsonb,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
            )
            """,
        )
    )
    conn.execute(
        text(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conname = 'uq_evaluation_artifacts_job_key'
                      AND conrelid = 'evaluation_artifacts'::regclass
                ) THEN
                    ALTER TABLE evaluation_artifacts
                    ADD CONSTRAINT uq_evaluation_artifacts_job_key UNIQUE (job_id, key);
                END IF;
            END $$;
            """,
        )
    )
    for ddl in (
        "CREATE INDEX IF NOT EXISTS ix_evaluation_artifacts_job_id ON evaluation_artifacts (job_id)",
        "CREATE INDEX IF NOT EXISTS ix_evaluation_artifacts_commit_hash ON evaluation_artifacts (commit_hash)",
        "CREATE INDEX IF NOT EXISTS ix_evaluation_artifacts_commit_card_id ON evaluation_artifacts (commit_card_id)",
        """
        CREATE INDEX IF NOT EXISTS ix_evaluation_artifacts_visibility_projection
        ON evaluation_artifacts (visibility, agent_projection)
        """,
    ):
        conn.execute(text(ddl))


MIGRATION = SchemaMigration(
    from_version=5,
    to_version=6,
    name="v0006_evaluation_artifacts",
    upgrade=upgrade,
)

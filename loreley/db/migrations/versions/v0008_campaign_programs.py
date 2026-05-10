from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from loreley.config import Settings
from loreley.db.migrations.registry import SchemaMigration


def upgrade(conn: Connection, _settings: Settings) -> None:
    """Create campaign program snapshots and nullable campaign hash columns."""

    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS campaign_programs (
                hash VARCHAR(64) PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                source_path VARCHAR(1024) NOT NULL,
                title VARCHAR(256),
                raw_markdown TEXT NOT NULL,
                normalized_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
                recognized_sections VARCHAR(64)[] NOT NULL DEFAULT ARRAY[]::VARCHAR(64)[],
                parse_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
            )
            """,
        )
    )
    for ddl in (
        "ALTER TABLE evolution_jobs ADD COLUMN IF NOT EXISTS campaign_program_hash VARCHAR(64)",
        "ALTER TABLE candidate_commits ADD COLUMN IF NOT EXISTS campaign_program_hash VARCHAR(64)",
        "ALTER TABLE evaluation_attempts ADD COLUMN IF NOT EXISTS campaign_program_hash VARCHAR(64)",
        "CREATE INDEX IF NOT EXISTS ix_evolution_jobs_campaign_program_hash ON evolution_jobs (campaign_program_hash)",
        "CREATE INDEX IF NOT EXISTS ix_candidate_commits_campaign_program_hash ON candidate_commits (campaign_program_hash)",
        """
        CREATE INDEX IF NOT EXISTS ix_evaluation_attempts_campaign_program_hash
        ON evaluation_attempts (campaign_program_hash)
        """,
    ):
        conn.execute(text(ddl))


MIGRATION = SchemaMigration(
    from_version=7,
    to_version=8,
    name="v0008_campaign_programs",
    upgrade=upgrade,
)

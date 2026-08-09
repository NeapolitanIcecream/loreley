from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from loreley.config import Settings
from loreley.db.migrations.registry import SchemaMigration


def upgrade(conn: Connection, settings: Settings) -> None:
    """Make evaluator attempts and their artifact evidence append-only."""

    del settings
    conn.execute(
        text(
            """
            ALTER TABLE evaluation_attempts
                ADD COLUMN IF NOT EXISTS run_token UUID,
                ADD COLUMN IF NOT EXISTS attempt_ordinal INTEGER,
                ADD COLUMN IF NOT EXISTS artifact_paths JSONB NOT NULL DEFAULT '{}'::jsonb
            """
        )
    )
    conn.execute(
        text(
            """
            WITH ranked AS (
                SELECT
                    id,
                    row_number() OVER (
                        PARTITION BY job_id
                        ORDER BY
                            started_at NULLS LAST,
                            created_at,
                            id
                    ) AS ordinal
                FROM evaluation_attempts
                WHERE job_id IS NOT NULL
            )
            UPDATE evaluation_attempts AS attempt
            SET attempt_ordinal = ranked.ordinal
            FROM ranked
            WHERE attempt.id = ranked.id
              AND attempt.attempt_ordinal IS NULL
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_evaluation_attempts_job_ordinal
                ON evaluation_attempts (job_id, attempt_ordinal)
                WHERE job_id IS NOT NULL AND attempt_ordinal IS NOT NULL
            """
        )
    )
    conn.execute(
        text(
            """
            ALTER TABLE evaluation_artifacts
                DROP CONSTRAINT IF EXISTS uq_evaluation_artifacts_job_key
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_evaluation_artifacts_attempt_key
                ON evaluation_artifacts (evaluation_attempt_id, key)
                WHERE evaluation_attempt_id IS NOT NULL
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_evaluation_artifacts_legacy_job_key
                ON evaluation_artifacts (job_id, key)
                WHERE evaluation_attempt_id IS NULL
            """
        )
    )


MIGRATION = SchemaMigration(
    from_version=20,
    to_version=21,
    name="v0021_append_only_evaluation_evidence",
    upgrade=upgrade,
)

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from loreley.config import Settings
from loreley.db.migrations.registry import SchemaMigration


def upgrade(conn: Connection, settings: Settings) -> None:
    """Persist evaluator-scoped candidate identities for archive deduplication."""

    del settings
    conn.execute(
        text(
            """
            ALTER TABLE candidate_commits
                ADD COLUMN IF NOT EXISTS candidate_identity VARCHAR(512),
                ADD COLUMN IF NOT EXISTS evaluation_identity_key VARCHAR(64)
            """
        )
    )
    conn.execute(
        text(
            """
            ALTER TABLE evaluation_attempts
                ADD COLUMN IF NOT EXISTS candidate_identity VARCHAR(512),
                ADD COLUMN IF NOT EXISTS evaluation_identity_key VARCHAR(64)
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS ix_candidate_commits_evaluation_identity_key
            ON candidate_commits (evaluation_identity_key)
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS ix_evaluation_attempts_identity_key
            ON evaluation_attempts (evaluation_identity_key)
            """
        )
    )


MIGRATION = SchemaMigration(
    from_version=15,
    to_version=16,
    name="v0016_candidate_identities",
    upgrade=upgrade,
)

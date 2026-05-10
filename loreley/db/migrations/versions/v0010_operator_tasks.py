from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from loreley.config import Settings
from loreley.db.migrations.registry import SchemaMigration


def upgrade(conn: Connection, _settings: Settings) -> None:
    """Create operator task records."""

    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS operator_tasks (
                id UUID PRIMARY KEY,
                kind VARCHAR(64) NOT NULL,
                status VARCHAR(32) NOT NULL DEFAULT 'pending',
                request_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                result_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                error_summary TEXT,
                started_at TIMESTAMP WITH TIME ZONE,
                completed_at TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
            )
            """,
        )
    )
    for ddl in (
        """
        CREATE INDEX IF NOT EXISTS ix_operator_tasks_kind_status_created
        ON operator_tasks (kind, status, created_at)
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_operator_tasks_status_started
        ON operator_tasks (status, started_at)
        """,
    ):
        conn.execute(text(ddl))


MIGRATION = SchemaMigration(
    from_version=9,
    to_version=10,
    name="v0010_operator_tasks",
    upgrade=upgrade,
)

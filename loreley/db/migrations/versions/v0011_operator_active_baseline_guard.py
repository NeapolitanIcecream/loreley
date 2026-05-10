from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from loreley.config import Settings
from loreley.db.migrations.registry import SchemaMigration


def upgrade(conn: Connection, _settings: Settings) -> None:
    """Add the active baseline operator task uniqueness guard."""

    conn.execute(
        text(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_operator_tasks_active_baseline_ensure
            ON operator_tasks (kind)
            WHERE kind = 'baseline_ensure' AND status IN ('pending', 'running')
            """,
        )
    )


MIGRATION = SchemaMigration(
    from_version=10,
    to_version=11,
    name="v0011_operator_active_baseline_guard",
    upgrade=upgrade,
)

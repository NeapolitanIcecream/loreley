from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from loreley.config import Settings
from loreley.db.migrations.registry import SchemaMigration


def upgrade(conn: Connection, settings: Settings) -> None:
    """Preserve the worker's complete bounded coding summary."""

    del settings
    conn.execute(
        text(
            """
            ALTER TABLE commit_cards
                ALTER COLUMN change_summary TYPE VARCHAR(800)
            """
        )
    )


MIGRATION = SchemaMigration(
    from_version=17,
    to_version=18,
    name="v0018_full_change_summaries",
    upgrade=upgrade,
)

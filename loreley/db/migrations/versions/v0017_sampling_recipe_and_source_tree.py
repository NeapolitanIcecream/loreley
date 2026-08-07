from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from loreley.config import Settings
from loreley.db.migrations.registry import SchemaMigration


def upgrade(conn: Connection, settings: Settings) -> None:
    """Persist restart-stable recipes and exact source-tree identities."""

    del settings
    conn.execute(
        text(
            """
            ALTER TABLE evolution_jobs
                ADD COLUMN IF NOT EXISTS sampling_ordinal INTEGER,
                ADD COLUMN IF NOT EXISTS sampling_recipe_hash VARCHAR(64),
                ADD COLUMN IF NOT EXISTS sampling_recipe_reused BOOLEAN NOT NULL DEFAULT FALSE
            """
        )
    )
    conn.execute(
        text(
            """
            ALTER TABLE candidate_commits
                ADD COLUMN IF NOT EXISTS source_tree_hash VARCHAR(64)
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS ix_evolution_jobs_island_recipe_created
            ON evolution_jobs (island_id, sampling_recipe_hash, created_at)
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_evolution_jobs_island_sampling_ordinal
            ON evolution_jobs (island_id, sampling_ordinal)
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS ix_candidate_commits_source_tree_contract
            ON candidate_commits (
                source_tree_hash,
                campaign_program_hash,
                evaluation_status
            )
            """
        )
    )


MIGRATION = SchemaMigration(
    from_version=16,
    to_version=17,
    name="v0017_sampling_recipe_and_source_tree",
    upgrade=upgrade,
)

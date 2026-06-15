from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from loreley.config import Settings
from loreley.db.migrations.registry import SchemaMigration


def upgrade(conn: Connection, _settings: Settings) -> None:
    """Create embedding cache compatibility manifests."""

    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache_manifests (
                id UUID PRIMARY KEY,
                cache_kind VARCHAR(64) NOT NULL,
                fingerprint VARCHAR(64) NOT NULL,
                payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                source VARCHAR(64) NOT NULL,
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
                    WHERE conname = 'uq_embedding_cache_manifests_cache_kind'
                      AND conrelid = 'embedding_cache_manifests'::regclass
                ) THEN
                    ALTER TABLE embedding_cache_manifests
                    ADD CONSTRAINT uq_embedding_cache_manifests_cache_kind UNIQUE (cache_kind);
                END IF;
            END $$;
            """,
        )
    )


MIGRATION = SchemaMigration(
    from_version=13,
    to_version=14,
    name="v0014_embedding_cache_manifests",
    upgrade=upgrade,
)

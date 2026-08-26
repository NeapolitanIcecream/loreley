from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from loreley.config import Settings
from loreley.db.migrations.registry import SchemaMigration


def upgrade(conn: Connection, settings: Settings) -> None:
    """Add the append-only evolution timeline and its migration boundary."""

    del settings
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS evolution_events (
                id UUID PRIMARY KEY,
                event_key VARCHAR(255) NOT NULL,
                event_type VARCHAR(64) NOT NULL,
                job_id UUID REFERENCES evolution_jobs(id) ON DELETE SET NULL,
                run_token UUID,
                island_id VARCHAR(64),
                commit_hash VARCHAR(64),
                occurred_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                ordinal INTEGER,
                duration_seconds DOUBLE PRECISION,
                payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                CONSTRAINT uq_evolution_events_event_key UNIQUE (event_key),
                CONSTRAINT ck_evolution_events_positive_ordinal
                    CHECK (ordinal IS NULL OR ordinal > 0),
                CONSTRAINT ck_evolution_events_nonnegative_duration
                    CHECK (duration_seconds IS NULL OR duration_seconds >= 0)
            )
            """
        )
    )
    for statement in (
        (
            "CREATE INDEX IF NOT EXISTS ix_evolution_events_order "
            "ON evolution_events (occurred_at, id)"
        ),
        (
            "CREATE INDEX IF NOT EXISTS ix_evolution_events_job_timeline "
            "ON evolution_events (job_id, occurred_at, id)"
        ),
        (
            "CREATE INDEX IF NOT EXISTS ix_evolution_events_type_timeline "
            "ON evolution_events (event_type, occurred_at, id)"
        ),
    ):
        conn.execute(text(statement))

    # This is a boundary observation, not reconstructed history.  It lets the
    # exporter distinguish pre-v23 state from events Loreley can prove exactly.
    conn.execute(
        text(
            """
            INSERT INTO evolution_events (
                id,
                event_key,
                event_type,
                occurred_at,
                payload
            )
            VALUES (
                md5('loreley:v23:timeline-history-boundary')::uuid,
                'v23:timeline-history-boundary',
                'timeline.history_boundary',
                NOW(),
                jsonb_build_object(
                    'reason', 'schema_migration',
                    'schema_from', 22,
                    'schema_to', 23
                )
            )
            ON CONFLICT (event_key) DO NOTHING
            """
        )
    )
    conn.execute(
        text(
            """
            INSERT INTO evolution_events (
                id,
                event_key,
                event_type,
                island_id,
                commit_hash,
                occurred_at,
                payload
            )
            SELECT
                md5(
                    'loreley:v23:archive-initial:' || island_id || ':' ||
                    cell_index::text || ':' || commit_hash
                )::uuid,
                'v23:archive-initial:' || md5(
                    island_id || ':' || cell_index::text || ':' || commit_hash
                ),
                'archive.member.initial_state',
                island_id,
                commit_hash,
                NOW(),
                jsonb_build_object(
                    'cell_index', cell_index,
                    'reason', 'migration_boundary'
                )
            FROM map_elites_archive_cells
            ON CONFLICT (event_key) DO NOTHING
            """
        )
    )


MIGRATION = SchemaMigration(
    from_version=22,
    to_version=23,
    name="v0023_evolution_events",
    upgrade=upgrade,
)

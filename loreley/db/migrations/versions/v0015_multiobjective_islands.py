from __future__ import annotations

import json

from sqlalchemy import text
from sqlalchemy.engine import Connection

from loreley.config import Settings, resolve_objective_contract
from loreley.db.migrations.registry import SchemaMigration


def upgrade(conn: Connection, settings: Settings) -> None:
    """Replace the scalar archive with Pareto fronts and add migration lineage."""

    contract = resolve_objective_contract(settings)
    conn.execute(
        text(
            """
            ALTER TABLE evolution_jobs
                ADD COLUMN IF NOT EXISTS migration_source_island_id VARCHAR(64),
                ADD COLUMN IF NOT EXISTS migration_commit_hash VARCHAR(64)
            """
        )
    )
    # A scalar winner per cell is insufficient input for a Pareto front. Drop
    # it and let normal ingestion rebuild from every durable successful result.
    conn.execute(text("DROP TABLE map_elites_archive_cells"))
    conn.execute(
        text(
            """
            CREATE TABLE map_elites_archive_cells (
                island_id VARCHAR(64) NOT NULL,
                cell_index INTEGER NOT NULL,
                commit_hash VARCHAR(64) NOT NULL,
                objective_values DOUBLE PRECISION[] NOT NULL
                    DEFAULT ARRAY[]::DOUBLE PRECISION[],
                measures DOUBLE PRECISION[] NOT NULL
                    DEFAULT ARRAY[]::DOUBLE PRECISION[],
                timestamp DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                PRIMARY KEY (island_id, cell_index, commit_hash),
                CONSTRAINT uq_map_elites_archive_island_commit
                    UNIQUE (island_id, commit_hash)
            )
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE INDEX ix_map_elites_archive_cells_commit_hash
            ON map_elites_archive_cells (commit_hash)
            """
        )
    )
    conn.execute(
        text(
            """
            UPDATE map_elites_states
            SET snapshot =
                (COALESCE(snapshot, '{}'::jsonb) - 'archive' - 'history')
                || jsonb_build_object(
                    'objective_contract',
                    CAST(:objective_contract AS JSONB),
                    'objective_contract_fingerprint',
                    CAST(:objective_contract_fingerprint AS TEXT)
                ),
                updated_at = now()
            """
        ),
        {
            "objective_contract": json.dumps(
                contract.as_payload(),
                separators=(",", ":"),
            ),
            "objective_contract_fingerprint": contract.fingerprint,
        },
    )
    conn.execute(
        text(
            """
            UPDATE evolution_jobs
            SET result_commit_hash = candidate_commit_hash
            WHERE status = 'SUCCEEDED'
              AND (result_commit_hash IS NULL OR result_commit_hash = '')
              AND candidate_commit_hash IS NOT NULL
              AND candidate_commit_hash <> ''
            """
        )
    )
    conn.execute(
        text(
            """
            UPDATE evolution_jobs
            SET ingestion_status = NULL,
                ingestion_attempts = 0,
                ingestion_delta = NULL,
                ingestion_status_code = NULL,
                ingestion_message = NULL,
                ingestion_cell_index = NULL,
                ingestion_last_attempt_at = NULL,
                ingestion_reason = NULL
            WHERE status = 'SUCCEEDED'
              AND result_commit_hash IS NOT NULL
              AND result_commit_hash <> ''
            """
        )
    )
    conn.execute(
        text(
            """
            UPDATE candidate_commits
            SET archive_status = 'not_considered',
                updated_at = now()
            WHERE evaluation_status = 'passed'
            """
        )
    )


MIGRATION = SchemaMigration(
    from_version=14,
    to_version=15,
    name="v0015_multiobjective_islands",
    upgrade=upgrade,
)

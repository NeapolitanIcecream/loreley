from __future__ import annotations

import uuid

from loguru import logger
from sqlalchemy import text
from sqlalchemy.engine import Connection

from loreley.config import Settings
from loreley.db.migrations.registry import SchemaMigration

log = logger.bind(module="db.migrations")


def _create_agent_actions(conn: Connection) -> None:
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS agent_actions (
                id UUID PRIMARY KEY,
                idempotency_key VARCHAR(256) NOT NULL DEFAULT '',
                actor VARCHAR(128) NOT NULL DEFAULT 'agent',
                action_type VARCHAR(64) NOT NULL,
                status VARCHAR(32) NOT NULL,
                dry_run BOOLEAN NOT NULL DEFAULT TRUE,
                request_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                expected_state JSONB NOT NULL DEFAULT '{}'::jsonb,
                result_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                error_code VARCHAR(64),
                error_summary TEXT,
                completed_at TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
            )
            """,
        )
    )
    for ddl in (
        """
        CREATE INDEX IF NOT EXISTS ix_agent_actions_action_created
        ON agent_actions (action_type, created_at)
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_agent_actions_status_created
        ON agent_actions (status, created_at)
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_actions_action_idempotency
        ON agent_actions (action_type, idempotency_key)
        WHERE idempotency_key <> ''
        """,
    ):
        conn.execute(text(ddl))


def _run_managed_index_cleanup(conn: Connection) -> None:
    for index_name in (
        "ix_commit_cards_commit_hash",
        "ix_map_elites_archive_cells_island",
        "ix_map_elites_repo_state_aggregates_commit",
    ):
        conn.execute(text(f'DROP INDEX IF EXISTS "{index_name}"'))
    for ddl in (
        """
        CREATE INDEX IF NOT EXISTS ix_evolution_jobs_ingestion_sort_expr
        ON evolution_jobs (
            status,
            ingestion_status,
            COALESCE(completed_at, created_at),
            id
        )
        WHERE result_commit_hash IS NOT NULL AND result_commit_hash <> ''
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_evolution_jobs_ui_sort_expr
        ON evolution_jobs (
            COALESCE(completed_at, created_at) DESC,
            id DESC
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_map_elites_archive_cells_island_commit
        ON map_elites_archive_cells (island_id, commit_hash)
        """,
    ):
        conn.execute(text(ddl))


def _backfill_candidate_commits(conn: Connection) -> None:
    rows = conn.execute(
        text(
            """
            WITH historical_jobs AS (
                SELECT
                    j.id,
                    COALESCE(NULLIF(j.result_commit_hash, ''), NULLIF(j.candidate_commit_hash, '')) AS commit_hash,
                    NULLIF(j.base_commit_hash, '') AS base_commit_hash,
                    j.island_id,
                    j.run_token,
                    j.job_kind,
                    j.candidate_branch_name,
                    j.candidate_published_at,
                    j.completed_at
                FROM evolution_jobs j
                WHERE lower(j.status::text) = 'succeeded'
            )
            SELECT
                h.id,
                h.commit_hash,
                h.base_commit_hash,
                h.island_id,
                h.run_token,
                h.job_kind,
                h.candidate_branch_name,
                h.candidate_published_at,
                h.completed_at,
                cc.id AS commit_card_id,
                EXISTS (
                    SELECT 1
                    FROM map_elites_archive_cells archive
                    WHERE archive.commit_hash = h.commit_hash
                ) AS archive_member
            FROM historical_jobs h
            LEFT JOIN commit_cards cc ON cc.commit_hash = h.commit_hash
            WHERE h.commit_hash IS NOT NULL
              AND h.base_commit_hash IS NOT NULL
            """,
        )
    ).mappings()

    inserted = 0
    skipped_existing = 0
    for row in rows:
        result = conn.execute(
            text(
                """
                INSERT INTO candidate_commits (
                    id,
                    commit_hash,
                    git_parent_commit_hash,
                    nearest_viable_ancestor_hash,
                    island_id,
                    produced_by_job_id,
                    run_token,
                    job_kind,
                    candidate_branch_name,
                    candidate_published_at,
                    publication_status,
                    evaluation_status,
                    archive_status,
                    lifecycle_status,
                    repair_state,
                    failed_depth,
                    repair_attempts,
                    repo_state_aggregate_status,
                    commit_card_id,
                    published_at,
                    evaluated_at,
                    archived_at
                )
                VALUES (
                    :id,
                    :commit_hash,
                    :git_parent_commit_hash,
                    :nearest_viable_ancestor_hash,
                    :island_id,
                    :produced_by_job_id,
                    :run_token,
                    :job_kind,
                    :candidate_branch_name,
                    :candidate_published_at,
                    :publication_status,
                    'passed',
                    :archive_status,
                    'active',
                    'audit_only',
                    0,
                    0,
                    'not_required',
                    :commit_card_id,
                    :published_at,
                    :evaluated_at,
                    :archived_at
                )
                ON CONFLICT (commit_hash) DO NOTHING
                """,
            ),
            {
                "id": uuid.uuid4(),
                "commit_hash": row["commit_hash"],
                "git_parent_commit_hash": row["base_commit_hash"],
                "nearest_viable_ancestor_hash": row["base_commit_hash"],
                "island_id": row["island_id"],
                "produced_by_job_id": row["id"],
                "run_token": row["run_token"],
                "job_kind": row["job_kind"] or "evolution",
                "candidate_branch_name": row["candidate_branch_name"],
                "candidate_published_at": row["candidate_published_at"],
                "publication_status": (
                    "published"
                    if row["candidate_branch_name"] or row["candidate_published_at"]
                    else "created"
                ),
                "archive_status": "member" if row["archive_member"] else "not_considered",
                "commit_card_id": row["commit_card_id"],
                "published_at": row["candidate_published_at"],
                "evaluated_at": row["completed_at"],
                "archived_at": row["completed_at"] if row["archive_member"] else None,
            },
        )
        if result.rowcount:
            inserted += int(result.rowcount)
        else:
            skipped_existing += 1

    log.info(
        "Best-effort candidate_commits backfill complete inserted={} skipped_existing={}",
        inserted,
        skipped_existing,
    )


def upgrade(conn: Connection, _settings: Settings) -> None:
    """Create agent action audit records and align managed indexes."""

    _create_agent_actions(conn)
    _run_managed_index_cleanup(conn)
    _backfill_candidate_commits(conn)


MIGRATION = SchemaMigration(
    from_version=11,
    to_version=12,
    name="v0012_agent_actions_and_cleanup",
    upgrade=upgrade,
)

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from loreley.config import Settings
from loreley.db.migrations.registry import SchemaMigration


def _add_evolution_job_columns(conn: Connection) -> None:
    conn.execute(text("ALTER TABLE evolution_jobs ADD COLUMN IF NOT EXISTS job_kind VARCHAR(32)"))
    conn.execute(
        text(
            """
            UPDATE evolution_jobs
            SET job_kind = CASE
                WHEN COALESCE(is_seed_job, FALSE) THEN 'seed'
                ELSE 'evolution'
            END
            WHERE job_kind IS NULL OR btrim(job_kind) = ''
            """,
        )
    )
    conn.execute(text("ALTER TABLE evolution_jobs ALTER COLUMN job_kind SET DEFAULT 'evolution'"))
    conn.execute(text("ALTER TABLE evolution_jobs ALTER COLUMN job_kind SET NOT NULL"))
    conn.execute(
        text("ALTER TABLE evolution_jobs ADD COLUMN IF NOT EXISTS repair_source_candidate_id UUID")
    )
    conn.execute(text("ALTER TABLE evolution_jobs ADD COLUMN IF NOT EXISTS repair_mode VARCHAR(32)"))


def _create_candidate_commits(conn: Connection) -> None:
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS candidate_commits (
                id UUID PRIMARY KEY,
                commit_hash VARCHAR(64) NOT NULL,
                git_parent_commit_hash VARCHAR(64) NOT NULL,
                nearest_viable_ancestor_hash VARCHAR(64),
                island_id VARCHAR(64),
                produced_by_job_id UUID NULL REFERENCES evolution_jobs(id) ON DELETE SET NULL,
                run_token UUID,
                job_kind VARCHAR(32) NOT NULL DEFAULT 'evolution',
                repair_source_candidate_id UUID,
                repair_mode VARCHAR(32),
                candidate_branch_name VARCHAR(255),
                candidate_published_at TIMESTAMP WITH TIME ZONE,
                publication_status VARCHAR(32) NOT NULL DEFAULT 'created',
                evaluation_status VARCHAR(32) NOT NULL DEFAULT 'not_evaluated',
                latest_evaluation_attempt_id UUID,
                archive_status VARCHAR(32) NOT NULL DEFAULT 'not_considered',
                lifecycle_status VARCHAR(32) NOT NULL DEFAULT 'active',
                failure_stage VARCHAR(32),
                failure_kind VARCHAR(64),
                failure_summary TEXT,
                failure_evidence_id UUID,
                repair_state VARCHAR(32) NOT NULL DEFAULT 'audit_only',
                failed_depth INTEGER NOT NULL DEFAULT 0,
                repair_attempts INTEGER NOT NULL DEFAULT 0,
                last_repair_job_id UUID NULL REFERENCES evolution_jobs(id) ON DELETE SET NULL,
                repo_state_aggregate_status VARCHAR(32) NOT NULL DEFAULT 'not_required',
                repo_state_aggregate_error TEXT,
                commit_card_id UUID NULL REFERENCES commit_cards(id) ON DELETE SET NULL,
                published_at TIMESTAMP WITH TIME ZONE,
                evaluated_at TIMESTAMP WITH TIME ZONE,
                archived_at TIMESTAMP WITH TIME ZONE,
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
                    WHERE conname = 'uq_candidate_commits_commit_hash'
                      AND conrelid = 'candidate_commits'::regclass
                ) THEN
                    ALTER TABLE candidate_commits
                    ADD CONSTRAINT uq_candidate_commits_commit_hash UNIQUE (commit_hash);
                END IF;
            END $$;
            """,
        )
    )


def _create_diagnostic_capsules(conn: Connection) -> None:
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS diagnostic_capsules (
                id UUID PRIMARY KEY,
                candidate_commit_id UUID NULL REFERENCES candidate_commits(id) ON DELETE CASCADE,
                job_id UUID NULL REFERENCES evolution_jobs(id) ON DELETE CASCADE,
                evaluation_attempt_id UUID,
                schema_version INTEGER NOT NULL DEFAULT 1,
                policy_version VARCHAR(64) NOT NULL,
                policy_passed BOOLEAN NOT NULL DEFAULT FALSE,
                payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                omitted_reasons VARCHAR(64)[] NOT NULL DEFAULT ARRAY[]::VARCHAR(64)[],
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
            )
            """,
        )
    )


def _create_evaluation_attempts(conn: Connection) -> None:
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS evaluation_attempts (
                id UUID PRIMARY KEY,
                candidate_commit_id UUID NULL REFERENCES candidate_commits(id) ON DELETE CASCADE,
                job_id UUID NULL REFERENCES evolution_jobs(id) ON DELETE CASCADE,
                evaluator_name VARCHAR(128),
                evaluator_version VARCHAR(128),
                outcome_kind VARCHAR(32) NOT NULL,
                failure_kind VARCHAR(64),
                failure_stage VARCHAR(32),
                repairability VARCHAR(32),
                safe_failure_summary TEXT,
                diagnostic_capsule_id UUID NULL REFERENCES diagnostic_capsules(id) ON DELETE SET NULL,
                artifact_policy_version VARCHAR(64),
                started_at TIMESTAMP WITH TIME ZONE,
                finished_at TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
            )
            """,
        )
    )


def _add_circular_constraints(conn: Connection) -> None:
    conn.execute(
        text(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'fk_evolution_jobs_repair_source_candidate_id'
                      AND conrelid = 'evolution_jobs'::regclass
                ) THEN
                    ALTER TABLE evolution_jobs
                    ADD CONSTRAINT fk_evolution_jobs_repair_source_candidate_id
                    FOREIGN KEY (repair_source_candidate_id)
                    REFERENCES candidate_commits(id)
                    ON DELETE SET NULL;
                END IF;

                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'fk_candidate_commits_latest_evaluation_attempt_id'
                      AND conrelid = 'candidate_commits'::regclass
                ) THEN
                    ALTER TABLE candidate_commits
                    ADD CONSTRAINT fk_candidate_commits_latest_evaluation_attempt_id
                    FOREIGN KEY (latest_evaluation_attempt_id)
                    REFERENCES evaluation_attempts(id);
                END IF;

                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'fk_candidate_commits_failure_evidence_id'
                      AND conrelid = 'candidate_commits'::regclass
                ) THEN
                    ALTER TABLE candidate_commits
                    ADD CONSTRAINT fk_candidate_commits_failure_evidence_id
                    FOREIGN KEY (failure_evidence_id)
                    REFERENCES diagnostic_capsules(id)
                    ON DELETE SET NULL;
                END IF;

                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'fk_candidate_commits_repair_source_candidate_id'
                      AND conrelid = 'candidate_commits'::regclass
                ) THEN
                    ALTER TABLE candidate_commits
                    ADD CONSTRAINT fk_candidate_commits_repair_source_candidate_id
                    FOREIGN KEY (repair_source_candidate_id)
                    REFERENCES candidate_commits(id)
                    ON DELETE SET NULL;
                END IF;

                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'fk_diagnostic_capsules_evaluation_attempt_id'
                      AND conrelid = 'diagnostic_capsules'::regclass
                ) THEN
                    ALTER TABLE diagnostic_capsules
                    ADD CONSTRAINT fk_diagnostic_capsules_evaluation_attempt_id
                    FOREIGN KEY (evaluation_attempt_id)
                    REFERENCES evaluation_attempts(id)
                    ON DELETE SET NULL;
                END IF;
            END $$;
            """,
        )
    )


def _create_indexes(conn: Connection) -> None:
    for ddl in (
        "CREATE INDEX IF NOT EXISTS ix_candidate_commits_produced_by_job_id ON candidate_commits (produced_by_job_id)",
        """
        CREATE INDEX IF NOT EXISTS ix_candidate_commits_repair_pool
        ON candidate_commits (island_id, repair_state, evaluation_status, updated_at)
        """,
        "CREATE INDEX IF NOT EXISTS ix_candidate_commits_repair_source ON candidate_commits (repair_source_candidate_id)",
        "CREATE INDEX IF NOT EXISTS ix_candidate_commits_git_parent ON candidate_commits (git_parent_commit_hash)",
        """
        CREATE INDEX IF NOT EXISTS ix_candidate_commits_nearest_viable_ancestor
        ON candidate_commits (nearest_viable_ancestor_hash)
        """,
        "CREATE INDEX IF NOT EXISTS ix_diagnostic_capsules_candidate_commit_id ON diagnostic_capsules (candidate_commit_id)",
        "CREATE INDEX IF NOT EXISTS ix_diagnostic_capsules_job_id ON diagnostic_capsules (job_id)",
        """
        CREATE INDEX IF NOT EXISTS ix_diagnostic_capsules_policy
        ON diagnostic_capsules (policy_version, policy_passed)
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_evaluation_attempts_candidate_started
        ON evaluation_attempts (candidate_commit_id, started_at)
        """,
        "CREATE INDEX IF NOT EXISTS ix_evaluation_attempts_job_id ON evaluation_attempts (job_id)",
        "CREATE INDEX IF NOT EXISTS ix_evaluation_attempts_outcome_kind ON evaluation_attempts (outcome_kind)",
        """
        CREATE INDEX IF NOT EXISTS ix_evolution_jobs_kind_status_scheduled
        ON evolution_jobs (job_kind, status, scheduled_at)
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_evolution_jobs_repair_source_status
        ON evolution_jobs (repair_source_candidate_id, status)
        """,
    ):
        conn.execute(text(ddl))


def upgrade(conn: Connection, _settings: Settings) -> None:
    """Create the repair-pool ledger and backfill required job kind state."""

    _add_evolution_job_columns(conn)
    _create_candidate_commits(conn)
    _create_diagnostic_capsules(conn)
    _create_evaluation_attempts(conn)
    _add_circular_constraints(conn)
    _create_indexes(conn)


MIGRATION = SchemaMigration(
    from_version=6,
    to_version=7,
    name="v0007_failed_candidate_repair_pool",
    upgrade=upgrade,
)

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from loreley.config import Settings
from loreley.db.migrations.registry import SchemaMigration


def upgrade(conn: Connection, _settings: Settings) -> None:
    """Create campaign baseline records."""

    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS campaign_baselines (
                id UUID PRIMARY KEY,
                baseline_key_hash VARCHAR(64) NOT NULL,
                root_commit_hash VARCHAR(64) NOT NULL,
                campaign_program_hash VARCHAR(64),
                evaluator_name VARCHAR(128),
                evaluator_version VARCHAR(128),
                primary_metric_name VARCHAR(128) NOT NULL,
                primary_metric_higher_is_better BOOLEAN NOT NULL DEFAULT TRUE,
                runtime_profile VARCHAR(128),
                effective_settings_fingerprint VARCHAR(64),
                status VARCHAR(32) NOT NULL,
                metric_value DOUBLE PRECISION,
                metric_unit VARCHAR(32),
                evaluation_summary TEXT,
                failure_kind VARCHAR(64),
                failure_summary TEXT,
                commit_card_id UUID NULL REFERENCES commit_cards(id) ON DELETE SET NULL,
                metric_id UUID NULL REFERENCES metrics(id) ON DELETE SET NULL,
                started_at TIMESTAMP WITH TIME ZONE,
                finished_at TIMESTAMP WITH TIME ZONE,
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
                    WHERE conname = 'uq_campaign_baselines_key_hash'
                      AND conrelid = 'campaign_baselines'::regclass
                ) THEN
                    ALTER TABLE campaign_baselines
                    ADD CONSTRAINT uq_campaign_baselines_key_hash UNIQUE (baseline_key_hash);
                END IF;
            END $$;
            """,
        )
    )
    for ddl in (
        "CREATE INDEX IF NOT EXISTS ix_campaign_baselines_root_commit ON campaign_baselines (root_commit_hash)",
        """
        CREATE INDEX IF NOT EXISTS ix_campaign_baselines_campaign_program_hash
        ON campaign_baselines (campaign_program_hash)
        """,
        "CREATE INDEX IF NOT EXISTS ix_campaign_baselines_status ON campaign_baselines (status)",
        "CREATE INDEX IF NOT EXISTS ix_campaign_baselines_commit_card_id ON campaign_baselines (commit_card_id)",
        "CREATE INDEX IF NOT EXISTS ix_campaign_baselines_metric_id ON campaign_baselines (metric_id)",
    ):
        conn.execute(text(ddl))


MIGRATION = SchemaMigration(
    from_version=8,
    to_version=9,
    name="v0009_campaign_baselines",
    upgrade=upgrade,
)

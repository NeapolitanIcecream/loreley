from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

from loreley.config import Settings
from loreley.db.migrations.registry import SchemaMigration


def upgrade(conn: Connection, _settings: Settings) -> None:
    """Create the LLM usage ledger."""

    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS llm_usage_events (
                id UUID PRIMARY KEY,
                job_id UUID NULL REFERENCES evolution_jobs(id) ON DELETE SET NULL,
                run_token UUID,
                phase VARCHAR(64) NOT NULL DEFAULT '',
                source VARCHAR(64) NOT NULL,
                provider VARCHAR(64) NOT NULL DEFAULT '',
                model VARCHAR(128) NOT NULL DEFAULT '',
                api_surface VARCHAR(64) NOT NULL DEFAULT '',
                input_tokens BIGINT NOT NULL DEFAULT 0,
                cached_input_tokens BIGINT NOT NULL DEFAULT 0,
                cache_write_tokens BIGINT NOT NULL DEFAULT 0,
                output_tokens BIGINT NOT NULL DEFAULT 0,
                reasoning_output_tokens BIGINT NOT NULL DEFAULT 0,
                total_tokens BIGINT NOT NULL DEFAULT 0,
                cost_usd NUMERIC(18, 8),
                cost_source VARCHAR(32) NOT NULL DEFAULT 'unpriced',
                pricing_version VARCHAR(128) NOT NULL DEFAULT '',
                raw_usage JSONB NOT NULL DEFAULT '{}'::jsonb,
                external_usage_id VARCHAR(256) NOT NULL DEFAULT '',
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
            )
            """,
        )
    )
    for ddl in (
        """
        CREATE INDEX IF NOT EXISTS ix_llm_usage_events_job_created
        ON llm_usage_events (job_id, created_at)
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_llm_usage_events_run_token
        ON llm_usage_events (run_token)
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_llm_usage_events_source_created
        ON llm_usage_events (source, created_at)
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_llm_usage_events_phase_created
        ON llm_usage_events (phase, created_at)
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_llm_usage_events_model_created
        ON llm_usage_events (model, created_at)
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_llm_usage_events_external_usage_id
        ON llm_usage_events (external_usage_id)
        WHERE external_usage_id <> ''
        """,
    ):
        conn.execute(text(ddl))


MIGRATION = SchemaMigration(
    from_version=12,
    to_version=13,
    name="v0013_llm_usage_events",
    upgrade=upgrade,
)

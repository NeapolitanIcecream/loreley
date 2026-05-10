from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from loguru import logger
from rich.console import Console
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.engine import URL
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from loreley.config import Settings, get_settings
from loreley.db.instance import ensure_instance_marker

INSTANCE_SCHEMA_VERSION = 12
_REDUNDANT_INDEX_NAMES = (
    "ix_commit_cards_commit_hash",
    "ix_map_elites_archive_cells_island",
    "ix_map_elites_repo_state_aggregates_commit",
)
_MANAGED_INDEX_DDL = (
    """
    CREATE INDEX IF NOT EXISTS "ix_evolution_jobs_campaign_program_hash"
    ON evolution_jobs (campaign_program_hash)
    """,
    """
    CREATE INDEX IF NOT EXISTS "ix_evaluation_attempts_campaign_program_hash"
    ON evaluation_attempts (campaign_program_hash)
    """,
    """
    CREATE INDEX IF NOT EXISTS "ix_candidate_commits_campaign_program_hash"
    ON candidate_commits (campaign_program_hash)
    """,
    """
    CREATE INDEX IF NOT EXISTS "ix_evolution_jobs_ingestion_sort_expr"
    ON evolution_jobs (
        status,
        ingestion_status,
        COALESCE(completed_at, created_at),
        id
    )
    WHERE result_commit_hash IS NOT NULL AND result_commit_hash <> ''
    """,
    """
    CREATE INDEX IF NOT EXISTS "ix_evolution_jobs_ui_sort_expr"
    ON evolution_jobs (
        COALESCE(completed_at, created_at) DESC,
        id DESC
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS "ix_map_elites_archive_cells_island_commit"
    ON map_elites_archive_cells (
        island_id,
        commit_hash
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS "uq_operator_tasks_active_baseline_ensure"
    ON operator_tasks (kind)
    WHERE kind = 'baseline_ensure' AND status IN ('pending', 'running')
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS "uq_agent_actions_action_idempotency"
    ON agent_actions (action_type, idempotency_key)
    WHERE idempotency_key <> ''
    """,
)

console = Console()
log = logger.bind(module="db.base")


def _sanitize_dsn(raw_dsn: str) -> str:
    """Hide sensitive parts of the DSN when logging."""
    url: URL = make_url(raw_dsn)
    if url.password:
        url = url.set(password="***")
    return str(url)


@lru_cache
def get_engine() -> Engine:
    """Create (and cache) the global SQLAlchemy engine."""
    settings = get_settings()
    engine = create_engine(
        settings.database_dsn,
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        echo=settings.db_echo,
        future=True,
    )
    safe_dsn = _sanitize_dsn(settings.database_dsn)
    console.log(f"[bold cyan]SQLAlchemy engine ready[/] {safe_dsn}")
    log.info("SQLAlchemy engine initialised for {}", safe_dsn)
    return engine


@lru_cache
def _session_factory() -> sessionmaker[Session]:
    """Create (and cache) the Session factory bound to the global engine."""
    return sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=get_engine(),
        expire_on_commit=False,
    )


class Base(DeclarativeBase):
    """Declarative base for ORM models."""

    pass


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope for DB operations."""
    session = _session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        log.exception("Session rollback triggered")
        raise
    finally:
        session.close()


def _run_managed_post_schema_ddl(engine: Engine) -> None:
    """Apply idempotent post-schema index cleanup and managed indexes."""

    with engine.begin() as conn:
        for index_name in _REDUNDANT_INDEX_NAMES:
            conn.execute(text(f'DROP INDEX IF EXISTS "{index_name}"'))
        for ddl in _MANAGED_INDEX_DDL:
            conn.execute(text(ddl))


def ensure_database_schema(
    *,
    validate_marker: bool = True,
    settings: Settings | None = None,
    auto_migrate: bool | None = None,
) -> None:
    """Ensure the Loreley database schema is initialized or migrated.

    Fresh databases are created at the current schema version when automatic
    migration is enabled. Existing Loreley databases with supported older
    schema markers are migrated before marker validation.
    """

    try:
        settings = settings or get_settings()
        # Import models so that all ORM tables are registered on ``Base.metadata``.
        import loreley.db.models  # noqa: F401  # pylint: disable=unused-import
        from loreley.db.migrations.runner import ensure_schema_current

        engine = get_engine()
        ensure_schema_current(
            engine=engine,
            settings=settings,
            target_version=INSTANCE_SCHEMA_VERSION,
            auto_migrate=settings.db_auto_migrate if auto_migrate is None else bool(auto_migrate),
        )
        _run_managed_post_schema_ddl(engine)
        if validate_marker:
            with session_scope() as session:
                ensure_instance_marker(
                    session=session,
                    settings=settings,
                    schema_version=INSTANCE_SCHEMA_VERSION,
                )
        console.log(
            "[green]Database schema ready[/] url={}".format(
                _sanitize_dsn(settings.database_dsn),
            ),
        )
        log.info("Database schema ensured for {}", _sanitize_dsn(settings.database_dsn))
    except Exception as exc:  # pragma: no cover - defensive
        console.log(
            "[bold red]Failed to ensure database schema[/] reason={}".format(exc),
        )
        log.exception("Failed to ensure database schema: {}", exc)
        raise


def reset_database_schema(*, include_console_log: bool = True) -> None:
    """Drop and recreate all Loreley ORM tables.

    This is a destructive local/disposable fallback. Normal upgrades should use
    ``uv run loreley db migrate`` so existing experiment data is preserved.

    Notes:
    - Uses `DROP TABLE ... CASCADE` to handle circular foreign key references.
    - This is destructive and should only be used for local/dev databases.
    """

    # Import models so that all ORM tables are registered on ``Base.metadata``.
    import loreley.db.models  # noqa: F401  # pylint: disable=unused-import

    settings = get_settings()
    safe_dsn = _sanitize_dsn(settings.database_dsn)
    if include_console_log:
        console.log(f"[bold yellow]Resetting database schema[/] url={safe_dsn}")
    log.warning("Resetting database schema (drop + create) url={}", safe_dsn)

    tables = list(Base.metadata.tables.values())
    engine = get_engine()
    with engine.begin() as conn:
        # Drop in reverse definition order; CASCADE makes the order resilient.
        for table in reversed(tables):
            name = table.name.replace('"', '""')
            conn.execute(text(f'DROP TABLE IF EXISTS "{name}" CASCADE'))
        conn.execute(text('DROP TABLE IF EXISTS "loreley_schema_migrations" CASCADE'))

    from loreley.db.migrations.runner import ensure_schema_current

    ensure_schema_current(
        engine=engine,
        settings=settings,
        target_version=INSTANCE_SCHEMA_VERSION,
        auto_migrate=True,
    )
    _run_managed_post_schema_ddl(engine)
    if include_console_log:
        console.log("[bold green]Database schema reset complete[/]")
    log.info("Database schema reset complete")

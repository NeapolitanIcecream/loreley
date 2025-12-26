from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from loguru import logger
from rich.console import Console
from sqlalchemy import text
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import DeclarativeBase, Session, scoped_session, sessionmaker

from loreley.config import get_settings

console = Console()
log = logger.bind(module="db.base")
settings = get_settings()


def _sanitize_dsn(raw_dsn: str) -> str:
    """Hide sensitive parts of the DSN when logging."""
    url: URL = make_url(raw_dsn)
    if url.password:
        url = url.set(password="***")
    return str(url)


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

SessionLocal = scoped_session(
    sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        expire_on_commit=False,
    ),
)


class Base(DeclarativeBase):
    """Declarative base for ORM models."""

    pass


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope for DB operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        log.exception("Session rollback triggered")
        raise
    finally:
        SessionLocal.remove()


def ensure_database_schema() -> None:
    """Ensure that all Loreley database tables exist.

    This helper imports the ORM models and issues ``CREATE TABLE IF NOT EXISTS``
    statements for all metadata. It is safe to call multiple times.
    """

    try:
        # Import models so that all ORM tables are registered on ``Base.metadata``.
        import loreley.db.models  # noqa: F401  # pylint: disable=unused-import

        Base.metadata.create_all(bind=engine)
        console.log(
            "[green]Database schema ready[/] url={}".format(
                settings.database_dsn,
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

    Loreley intentionally does not ship migrations. For development workflows,
    the safest way to align the DB schema with the current ORM models is to
    drop all ORM-managed tables and recreate them.

    Notes:
    - Uses `DROP TABLE ... CASCADE` to handle circular foreign key references.
    - This is destructive and should only be used for local/dev databases.
    """

    # Import models so that all ORM tables are registered on ``Base.metadata``.
    import loreley.db.models  # noqa: F401  # pylint: disable=unused-import

    safe_dsn = _sanitize_dsn(settings.database_dsn)
    if include_console_log:
        console.log(f"[bold yellow]Resetting database schema[/] url={safe_dsn}")
    log.warning("Resetting database schema (drop + create) url={}", safe_dsn)

    tables = list(Base.metadata.tables.values())
    with engine.begin() as conn:
        # Drop in reverse definition order; CASCADE makes the order resilient.
        for table in reversed(tables):
            name = table.name.replace('"', '""')
            conn.execute(text(f'DROP TABLE IF EXISTS "{name}" CASCADE'))

    Base.metadata.create_all(bind=engine)
    if include_console_log:
        console.log("[bold green]Database schema reset complete[/]")
    log.info("Database schema reset complete")

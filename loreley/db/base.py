from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from loguru import logger
from rich.console import Console
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

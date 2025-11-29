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

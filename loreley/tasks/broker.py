from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import dramatiq
from dramatiq.brokers.redis import RedisBroker
from loguru import logger
from rich.console import Console

from loreley.config import Settings, get_settings

console = Console()
log = logger.bind(module="tasks.broker")

__all__ = ["broker", "setup_broker", "build_redis_broker"]


def _safe_connection_repr(settings: Settings) -> str:
    """Return a Redis connection representation that is safe to log.

    This deliberately omits any credentials that may be present in TASKS_REDIS_URL.
    """

    if settings.tasks_redis_url:
        parsed = urlparse(settings.tasks_redis_url)
        scheme = parsed.scheme or "redis"
        host = parsed.hostname or "localhost"
        port = f":{parsed.port}" if parsed.port is not None else ""
        # Redis DB index is typically encoded in the path, e.g. "/0".
        path = parsed.path or ""
        return f"{scheme}://{host}{port}{path}"

    return f"{settings.tasks_redis_host}:{settings.tasks_redis_port}/{settings.tasks_redis_db}"


def build_redis_broker(settings: Settings | None = None) -> RedisBroker:
    """Instantiate the Redis broker using application configuration."""

    settings = settings or get_settings()
    broker_kwargs: dict[str, Any] = {
        "namespace": settings.tasks_redis_namespace,
    }
    if settings.tasks_redis_url:
        broker_kwargs["url"] = settings.tasks_redis_url
    else:
        broker_kwargs.update(
            host=settings.tasks_redis_host,
            port=settings.tasks_redis_port,
            db=settings.tasks_redis_db,
        )
        if settings.tasks_redis_password:
            broker_kwargs["password"] = settings.tasks_redis_password
    return RedisBroker(**broker_kwargs)


def setup_broker(settings: Settings | None = None) -> RedisBroker:
    """Configure dramatiq to use the Redis broker."""

    settings = settings or get_settings()
    redis_broker = build_redis_broker(settings)
    dramatiq.set_broker(redis_broker)

    connection_repr = _safe_connection_repr(settings)
    console.log(
        "[bold green]Configured dramatiq broker[/] "
        f"redis={connection_repr} namespace={settings.tasks_redis_namespace!r}",
    )
    log.info(
        "Redis broker ready: redis={} namespace={}",
        connection_repr,
        settings.tasks_redis_namespace,
    )
    return redis_broker


broker = setup_broker()


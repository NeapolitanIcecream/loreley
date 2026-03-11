from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import dramatiq
from dramatiq.brokers.redis import RedisBroker
from loguru import logger
from rich.console import Console

from loreley.config import Settings, get_settings
from loreley.naming import tasks_redis_namespace

console = Console()
log = logger.bind(module="tasks.broker")

broker: RedisBroker | None = None

__all__ = ["broker", "setup_broker", "build_redis_broker", "reset_redis_namespace"]


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
    namespace = tasks_redis_namespace(settings.experiment_id)
    broker_kwargs: dict[str, Any] = {
        "namespace": namespace,
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

    global broker
    settings = settings or get_settings()
    redis_broker = build_redis_broker(settings)
    dramatiq.set_broker(redis_broker)
    broker = redis_broker

    connection_repr = _safe_connection_repr(settings)
    console.log(
        "[bold green]Configured dramatiq broker[/] "
        f"redis={connection_repr} namespace={redis_broker.namespace!r}",
    )
    log.info(
        "Redis broker ready: redis={} namespace={}",
        connection_repr,
        redis_broker.namespace,
    )
    return redis_broker


def reset_redis_namespace(settings: Settings | None = None) -> int:
    """Delete all Redis keys under the experiment namespace.

    `RedisBroker.flush_all()` only clears declared queues. During a DB reset we
    often have not declared any queues yet, so experiment-scoped message lists
    can survive. This helper scans the namespace directly to guarantee a clean
    slate for the next run.
    """

    redis_broker = build_redis_broker(settings)
    pattern = f"{redis_broker.namespace}:*"
    keys = list(redis_broker.client.scan_iter(match=pattern))
    if not keys:
        return 0
    deleted = int(redis_broker.client.delete(*keys) or 0)
    log.info(
        "Redis namespace reset: namespace={} deleted_keys={}",
        redis_broker.namespace,
        deleted,
    )
    return deleted

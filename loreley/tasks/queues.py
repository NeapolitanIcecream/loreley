"""Queue naming helpers for Dramatiq routing.

Loreley uses one queue per experiment to keep worker processes experiment-attached
and to avoid runtime configuration drift across experiments.
"""

from __future__ import annotations

import uuid

DEFAULT_TASKS_QUEUE_PREFIX = "loreley.evolution"


def _coerce_uuid(value: uuid.UUID | str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def experiment_queue_name(*, base_queue: str | None, experiment_id: uuid.UUID | str) -> str:
    """Return a stable per-experiment queue name.

    The queue name is derived as:
        "{base_queue}.{experiment_id.hex}"
    where `base_queue` defaults to `DEFAULT_TASKS_QUEUE_PREFIX`.
    """

    base = (base_queue or "").strip() or DEFAULT_TASKS_QUEUE_PREFIX
    exp = _coerce_uuid(experiment_id)
    # Idempotency: if the caller already provided the full derived queue name,
    # do not append the suffix twice.
    suffix = f".{exp.hex}"
    if base.endswith(suffix):
        return base
    return f"{base}{suffix}"


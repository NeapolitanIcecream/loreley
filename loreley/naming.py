"""Naming helpers derived from EXPERIMENT_ID.

Loreley assumes "one experiment per database" for operational isolation.
To avoid cross-experiment interference on shared infrastructure (Redis keys,
queue names, filesystem paths, remote branch patterns), all such names are
derived from a single env input: EXPERIMENT_ID.

EXPERIMENT_ID can be either:
- a UUID string (recommended for fully opaque identifiers), or
- a short human-friendly slug (e.g. "circlepack-jan21").

When a UUID is required internally (e.g. DB primary keys), slugs are mapped to
UUIDs using uuid5 with a fixed namespace so the mapping is stable.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from typing import Any

EXPERIMENT_ID_ENV: str = "EXPERIMENT_ID"

# Base names used for derived identifiers. These are intentionally not configurable
# to keep the operational model simple and consistent.
DEFAULT_TASKS_REDIS_NAMESPACE_PREFIX: str = "loreley"
DEFAULT_TASKS_QUEUE_PREFIX: str = "loreley.evolution"
DEFAULT_WORKER_JOB_BRANCH_PREFIX: str = "evolution/job"

# Fixed namespace for uuid5 mapping of slug experiment ids.
#
# Note: use a fixed namespace so slug -> uuid mapping is stable across processes
# and machines without requiring any shared registry.
_UUID5_NAMESPACE: uuid.UUID = uuid.uuid5(uuid.NAMESPACE_DNS, "ace.loreley.experiment_id")

_SAFE_NAMESPACE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_DASH_COLLAPSE_RE = re.compile(r"-{2,}")


@dataclass(frozen=True, slots=True)
class ExperimentIdentity:
    """Derived experiment identity information."""

    raw: str
    namespace: str
    uuid: uuid.UUID


def _clean_raw(value: uuid.UUID | str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _try_parse_uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value))
    except Exception:
        return None


def _slugify(value: str) -> str:
    """Return a filesystem/queue/branch-safe slug for the provided string."""
    text = (value or "").strip()
    if not text:
        return ""
    replaced = _SAFE_NAMESPACE_RE.sub("-", text)
    collapsed = _DASH_COLLAPSE_RE.sub("-", replaced).strip("-")
    return collapsed.lower()


def _short_hash(value: str, *, length: int = 8) -> str:
    raw = (value or "").encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[: max(4, int(length))]


def resolve_experiment_identity(experiment_id: uuid.UUID | str | None) -> ExperimentIdentity:
    """Resolve experiment identity from a user-supplied EXPERIMENT_ID."""
    raw = _clean_raw(experiment_id)
    if raw is None:
        raise ValueError(f"{EXPERIMENT_ID_ENV} is required.")

    parsed = _try_parse_uuid(raw)
    if parsed is not None:
        namespace = parsed.hex
        return ExperimentIdentity(raw=raw, namespace=namespace, uuid=parsed)

    mapped = uuid.uuid5(_UUID5_NAMESPACE, raw)
    base = _slugify(raw) or "exp"
    suffix = mapped.hex[:8]
    namespace = f"{base}-{suffix}"
    return ExperimentIdentity(raw=raw, namespace=namespace, uuid=mapped)


def resolve_experiment_namespace(experiment_id: uuid.UUID | str | None) -> str:
    """Return the derived experiment namespace string."""
    return resolve_experiment_identity(experiment_id).namespace


def resolve_experiment_uuid(experiment_id: uuid.UUID | str | None) -> uuid.UUID:
    """Return the derived experiment UUID (uuid5 mapping for slugs)."""
    return resolve_experiment_identity(experiment_id).uuid


def tasks_redis_namespace(experiment_id: uuid.UUID | str | None) -> str:
    """Return the Redis broker namespace for the experiment."""
    ns = resolve_experiment_namespace(experiment_id)
    return f"{DEFAULT_TASKS_REDIS_NAMESPACE_PREFIX}.{ns}"


def tasks_queue_name(experiment_id: uuid.UUID | str | None) -> str:
    """Return the Dramatiq queue name for the experiment."""
    ns = resolve_experiment_namespace(experiment_id)
    return f"{DEFAULT_TASKS_QUEUE_PREFIX}.{ns}"


def worker_job_branch_prefix(experiment_id: uuid.UUID | str | None) -> str:
    """Return the remote job branch prefix for the experiment."""
    ns = resolve_experiment_namespace(experiment_id)
    return f"{DEFAULT_WORKER_JOB_BRANCH_PREFIX}/{ns}"


def safe_namespace_or_none(experiment_id: uuid.UUID | str | None) -> str | None:
    """Best-effort experiment namespace without raising.

    This is useful for optional contexts (e.g. UI/API processes) where
    EXPERIMENT_ID may not be configured.
    """
    raw = _clean_raw(experiment_id)
    if raw is None:
        return None
    parsed = _try_parse_uuid(raw)
    if parsed is not None:
        return parsed.hex
    mapped = uuid.uuid5(_UUID5_NAMESPACE, raw)
    base = _slugify(raw) or "exp"
    return f"{base}-{mapped.hex[:8]}"


def safe_namespace_from_settings(settings: Any) -> str | None:
    """Best-effort namespace derived from a Settings-like object."""
    return safe_namespace_or_none(getattr(settings, "experiment_id", None))


__all__ = [
    "ExperimentIdentity",
    "DEFAULT_TASKS_QUEUE_PREFIX",
    "DEFAULT_TASKS_REDIS_NAMESPACE_PREFIX",
    "DEFAULT_WORKER_JOB_BRANCH_PREFIX",
    "EXPERIMENT_ID_ENV",
    "resolve_experiment_identity",
    "resolve_experiment_namespace",
    "resolve_experiment_uuid",
    "safe_namespace_from_settings",
    "safe_namespace_or_none",
    "tasks_queue_name",
    "tasks_redis_namespace",
    "worker_job_branch_prefix",
]


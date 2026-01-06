"""Experiment-scoped behaviour configuration helpers.

The scheduler persists an experiment config snapshot (JSONB) in the database.
Workers and the UI/API must interpret experiment data using that persisted
snapshot instead of relying on local environment variables.
"""

from __future__ import annotations

import uuid
from typing import Any, Mapping

from sqlalchemy import select

from loreley.config import Settings, get_settings
from loreley.db.base import session_scope
from loreley.db.models import Experiment

__all__ = [
    "ExperimentConfigError",
    "load_experiment_config_snapshot",
    "apply_experiment_config_snapshot",
    "resolve_experiment_settings",
]

BEHAVIOR_SNAPSHOT_PREFIXES = ("mapelites_", "worker_evaluator_")


class ExperimentConfigError(RuntimeError):
    """Raised when experiment configuration cannot be resolved from the DB."""


def _coerce_uuid(value: uuid.UUID | str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def _restore_json_compatible(value: Any) -> Any:
    """Restore non-finite float sentinels stored in JSONB snapshots."""

    if isinstance(value, Mapping):
        marker = value.get("__float__") if len(value) == 1 else None
        if isinstance(marker, str):
            if marker == "nan":
                return float("nan")
            if marker == "inf":
                return float("inf")
            if marker == "-inf":
                return float("-inf")
        return {str(k): _restore_json_compatible(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_json_compatible(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_restore_json_compatible(v) for v in value)
    return value


def load_experiment_config_snapshot(experiment_id: uuid.UUID | str) -> dict[str, Any]:
    """Load the persisted config snapshot for the given experiment id."""

    exp_id = _coerce_uuid(experiment_id)
    with session_scope() as session:
        stmt = select(Experiment.config_snapshot).where(Experiment.id == exp_id)
        snapshot = session.execute(stmt).scalar_one_or_none()
    if snapshot is None:
        raise ExperimentConfigError(f"Experiment not found: {exp_id}")
    return dict(snapshot or {})


def apply_experiment_config_snapshot(
    *,
    base_settings: Settings,
    snapshot: Mapping[str, Any],
) -> Settings:
    """Return Settings with experiment snapshot values applied over the base settings."""

    if not snapshot:
        return base_settings

    restored = _restore_json_compatible(snapshot)
    if not isinstance(restored, Mapping):
        raise ExperimentConfigError("Experiment config snapshot must be a mapping.")

    overrides = {
        str(k): v for k, v in restored.items() if str(k).startswith(BEHAVIOR_SNAPSHOT_PREFIXES)
    }
    if not overrides:
        return base_settings

    # NOTE: `Settings` inherits from `BaseSettings`, where environment variables may take
    # precedence over explicit constructor inputs. For experiment-scoped interpretation we
    # must ensure the persisted snapshot wins over the process environment, therefore we
    # apply overrides onto the already-loaded Settings instance.
    valid = {k: v for k, v in overrides.items() if k in type(base_settings).model_fields}
    return base_settings.model_copy(update=valid)


def resolve_experiment_settings(
    *,
    experiment_id: uuid.UUID | str,
    base_settings: Settings | None = None,
) -> Settings:
    """Convenience wrapper: load snapshot from DB and build effective settings."""

    base = base_settings or get_settings()
    snapshot = load_experiment_config_snapshot(experiment_id)
    return apply_experiment_config_snapshot(base_settings=base, snapshot=snapshot)



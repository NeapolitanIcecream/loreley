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
    "EXPERIMENT_SNAPSHOT_SCHEMA_VERSION",
    "experiment_behavior_keys",
    "ExperimentConfigError",
    "load_experiment_config_snapshot",
    "apply_experiment_config_snapshot",
    "resolve_experiment_settings",
]

EXPERIMENT_SNAPSHOT_SCHEMA_VERSION = 2

EXPERIMENT_BEHAVIOR_PREFIXES: tuple[str, ...] = (
    "mapelites_",
    "worker_evaluator_",
    "worker_evolution_",
    "worker_planning_",
    "worker_coding_",
    "worker_cursor_",
    "openai_",
)

# Experiment snapshots must not persist secrets or deployment-only wiring.
EXPERIMENT_BEHAVIOR_EXCLUDED_KEYS: frozenset[str] = frozenset(
    {
        # Secrets.
        "openai_api_key",
        # Deployment-scoped environment passthrough.
        "worker_planning_extra_env",
        "worker_coding_extra_env",
        # Deployment-scoped agent binaries and local schema paths.
        "worker_planning_codex_bin",
        "worker_planning_codex_profile",
        "worker_planning_schema_path",
        "worker_coding_codex_bin",
        "worker_coding_codex_profile",
        "worker_coding_schema_path",
    }
)


def _build_experiment_behavior_keys() -> tuple[str, ...]:
    keys: list[str] = []
    for name in Settings.model_fields:
        if not any(name.startswith(prefix) for prefix in EXPERIMENT_BEHAVIOR_PREFIXES):
            continue
        if name in EXPERIMENT_BEHAVIOR_EXCLUDED_KEYS:
            continue
        keys.append(name)
    keys_sorted = tuple(sorted(keys))
    if not keys_sorted:
        raise RuntimeError("Experiment behavior keyset must not be empty.")
    return keys_sorted


EXPERIMENT_BEHAVIOR_KEYS: tuple[str, ...] = _build_experiment_behavior_keys()


def experiment_behavior_keys() -> tuple[str, ...]:
    """Return the full set of experiment-scoped behaviour keys."""

    return EXPERIMENT_BEHAVIOR_KEYS

# Keys that are required in every persisted experiment snapshot. Loreley does not
# support forward-compatible snapshot schemas; missing keys indicate a stale DB.
_REQUIRED_SNAPSHOT_KEYS: tuple[str, ...] = (
    "experiment_snapshot_schema_version",
    *EXPERIMENT_BEHAVIOR_KEYS,
)


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


def _sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_experiment_snapshot(snapshot: Mapping[str, Any]) -> None:
    """Validate the persisted experiment snapshot.

    Loreley intentionally does not support forward compatibility for experiment
    config snapshots. When the snapshot schema does not match, operators should
    reset the database schema for development workflows.
    """

    missing = [key for key in _REQUIRED_SNAPSHOT_KEYS if key not in snapshot]
    if missing:
        preview = ", ".join(missing[:20])
        suffix = f", ... (+{len(missing) - 20} more)" if len(missing) > 20 else ""
        raise ExperimentConfigError(
            "Experiment config snapshot is missing required keys: "
            f"{preview}{suffix}. "
            "Loreley does not support forward-compatible snapshot schemas; "
            "reset the database schema to upgrade.",
        )

    raw_version = snapshot.get("experiment_snapshot_schema_version")
    if raw_version is None:
        version = -1
    else:
        try:
            version = int(raw_version)
        except (TypeError, ValueError):
            version = -1
    if version != EXPERIMENT_SNAPSHOT_SCHEMA_VERSION:
        raise ExperimentConfigError(
            "Experiment config snapshot schema version mismatch "
            f"(expected {EXPERIMENT_SNAPSHOT_SCHEMA_VERSION} got {raw_version!r}). "
            "Loreley does not support forward-compatible snapshot schemas; "
            "reset the database schema to upgrade.",
        )

    root_commit = str(snapshot.get("mapelites_experiment_root_commit") or "").strip()
    if not root_commit:
        raise ExperimentConfigError(
            "Experiment config snapshot is missing a non-empty mapelites_experiment_root_commit.",
        )

    ignore_text = snapshot.get("mapelites_repo_state_ignore_text")
    if not isinstance(ignore_text, str):
        raise ExperimentConfigError(
            "Experiment config snapshot has invalid mapelites_repo_state_ignore_text "
            f"(expected str got {type(ignore_text).__name__}).",
        )
    ignore_sha = snapshot.get("mapelites_repo_state_ignore_sha256")
    if not isinstance(ignore_sha, str) or not ignore_sha.strip():
        raise ExperimentConfigError(
            "Experiment config snapshot has invalid mapelites_repo_state_ignore_sha256 "
            f"(expected non-empty str got {ignore_sha!r}).",
        )
    expected_sha = _sha256_text(ignore_text)
    if ignore_sha.strip() != expected_sha:
        raise ExperimentConfigError(
            "Experiment config snapshot ignore hash mismatch "
            f"(expected {expected_sha} got {ignore_sha.strip()}). "
            "Reset the database schema if this snapshot was produced by an older Loreley version.",
        )


def load_experiment_config_snapshot(experiment_id: uuid.UUID | str) -> dict[str, Any]:
    """Load the persisted config snapshot for the given experiment id."""

    exp_id = _coerce_uuid(experiment_id)
    with session_scope() as session:
        stmt = select(Experiment.config_snapshot).where(Experiment.id == exp_id)
        snapshot = session.execute(stmt).scalar_one_or_none()
    if snapshot is None:
        raise ExperimentConfigError(f"Experiment not found: {exp_id}")
    payload = dict(snapshot or {})
    restored = _restore_json_compatible(payload)
    if not isinstance(restored, Mapping):
        raise ExperimentConfigError("Experiment config snapshot must be a mapping.")
    _validate_experiment_snapshot(restored)
    return dict(restored)


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

    allowed = set(EXPERIMENT_BEHAVIOR_KEYS)
    overrides = {str(k): v for k, v in restored.items() if str(k) in allowed}
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



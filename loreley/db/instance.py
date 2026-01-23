from __future__ import annotations

from dataclasses import dataclass
import uuid

from sqlalchemy.orm import Session

from loreley.config import Settings
from loreley.naming import resolve_experiment_identity

RESET_DB_HINT = "Reset the database schema with `uv run loreley reset-db --yes`."


class InstanceMetadataError(RuntimeError):
    """Raised when the instance metadata marker is missing or mismatched."""


@dataclass(frozen=True, slots=True)
class InstanceIdentity:
    """Resolved instance identity required for schema validation."""

    experiment_raw: str
    experiment_uuid: uuid.UUID
    root_commit: str


def resolve_instance_identity(settings: Settings) -> InstanceIdentity:
    """Resolve experiment identity and root commit from settings."""
    try:
        identity = resolve_experiment_identity(settings.experiment_id)
    except ValueError as exc:
        raise InstanceMetadataError(str(exc)) from exc

    root_commit = (settings.mapelites_experiment_root_commit or "").strip()
    if not root_commit:
        raise InstanceMetadataError("MAPELITES_EXPERIMENT_ROOT_COMMIT is required.")
    return InstanceIdentity(
        experiment_raw=identity.raw,
        experiment_uuid=identity.uuid,
        root_commit=root_commit,
    )


def root_commit_matches(stored: str, configured: str) -> bool:
    """Return True when two root commit hashes refer to the same commit."""
    stored = (stored or "").strip()
    configured = (configured or "").strip()
    if not stored or not configured:
        return False
    return stored.startswith(configured) or configured.startswith(stored)


def validate_instance_marker(
    *,
    session: Session,
    settings: Settings,
    schema_version: int,
) -> InstanceMetadata:
    """Validate the instance metadata marker and return the ORM row."""
    from loreley.db.models import InstanceMetadata

    identity = resolve_instance_identity(settings)

    meta = session.get(InstanceMetadata, 1)
    if meta is None:
        raise InstanceMetadataError(f"Instance metadata is missing. {RESET_DB_HINT}")
    if int(meta.schema_version or 0) != int(schema_version):
        raise InstanceMetadataError(
            f"Instance metadata schema_version mismatch. {RESET_DB_HINT}",
        )
    if str(meta.experiment_id_raw or "").strip() != identity.experiment_raw:
        raise InstanceMetadataError(
            f"EXPERIMENT_ID does not match the database marker. {RESET_DB_HINT}",
        )
    if str(meta.experiment_uuid or "") != str(identity.experiment_uuid):
        raise InstanceMetadataError(
            f"EXPERIMENT_ID UUID mapping does not match the database marker. {RESET_DB_HINT}",
        )
    meta_root = str(meta.root_commit_hash or "").strip()
    if not root_commit_matches(meta_root, identity.root_commit):
        raise InstanceMetadataError(
            "MAPELITES_EXPERIMENT_ROOT_COMMIT does not match the database marker. "
            f"{RESET_DB_HINT}",
        )
    return meta


def seed_instance_marker(
    *,
    session: Session,
    settings: Settings,
    schema_version: int,
) -> None:
    """Seed or update the instance metadata marker."""
    from loreley.db.models import InstanceMetadata

    identity = resolve_instance_identity(settings)
    meta = InstanceMetadata(
        id=1,
        schema_version=int(schema_version),
        experiment_id_raw=identity.experiment_raw,
        experiment_uuid=identity.experiment_uuid,
        root_commit_hash=identity.root_commit,
    )
    session.merge(meta)


__all__ = [
    "InstanceIdentity",
    "InstanceMetadataError",
    "RESET_DB_HINT",
    "resolve_instance_identity",
    "root_commit_matches",
    "seed_instance_marker",
    "validate_instance_marker",
]

from __future__ import annotations

from dataclasses import dataclass
import uuid

import pytest

from loreley.config import Settings
from loreley.db.base import INSTANCE_SCHEMA_VERSION
from loreley.db.instance import (
    InstanceMetadataError,
    resolve_instance_namespace_from_marker,
    root_commit_matches,
    seed_instance_marker,
    validate_instance_marker,
    validate_instance_marker_schema,
)
from loreley.naming import resolve_experiment_identity, safe_namespace_or_none


@dataclass
class FakeMeta:
    schema_version: int
    experiment_id_raw: str
    experiment_uuid: uuid.UUID
    root_commit_hash: str
    repository_slug: str | None = None
    repository_canonical_origin: str | None = None


class FakeSession:
    def __init__(self, meta: FakeMeta | None) -> None:
        self._meta = meta
        self.merged: object | None = None

    def get(self, _model, key: int):  # type: ignore[no-untyped-def]
        assert key == 1
        return self._meta

    def merge(self, value):  # type: ignore[no-untyped-def]
        self.merged = value


def _settings(exp: str = "demo", root: str = "deadbeef") -> Settings:
    return Settings(
        _env_file=None,
        experiment_id=exp,
        mapelites_experiment_root_commit=root,
    )


def _meta_for(exp: str, root: str, *, schema_version: int | None = None) -> FakeMeta:
    identity = resolve_experiment_identity(exp)
    return FakeMeta(
        schema_version=INSTANCE_SCHEMA_VERSION if schema_version is None else schema_version,
        experiment_id_raw=identity.raw,
        experiment_uuid=identity.uuid,
        root_commit_hash=root,
    )


def test_root_commit_matches_accepts_prefixes() -> None:
    assert root_commit_matches("deadbeef", "deadbeef00")
    assert root_commit_matches("deadbeef00", "deadbeef")


def test_validate_instance_marker_schema_requires_row() -> None:
    session = FakeSession(meta=None)
    with pytest.raises(InstanceMetadataError, match="Instance metadata is missing"):
        validate_instance_marker_schema(
            session=session,
            schema_version=INSTANCE_SCHEMA_VERSION,
        )


def test_validate_instance_marker_schema_requires_version_match() -> None:
    meta = _meta_for("demo", "deadbeef", schema_version=INSTANCE_SCHEMA_VERSION - 1)
    session = FakeSession(meta=meta)
    with pytest.raises(InstanceMetadataError, match="schema_version mismatch"):
        validate_instance_marker_schema(
            session=session,
            schema_version=INSTANCE_SCHEMA_VERSION,
        )


def test_validate_instance_marker_rejects_experiment_id_mismatch() -> None:
    settings = _settings(exp="demo", root="deadbeef")
    meta = _meta_for("other", "deadbeef")
    session = FakeSession(meta=meta)
    with pytest.raises(InstanceMetadataError, match="EXPERIMENT_ID does not match"):
        validate_instance_marker(
            session=session,
            settings=settings,
            schema_version=INSTANCE_SCHEMA_VERSION,
        )


def test_validate_instance_marker_rejects_uuid_mismatch() -> None:
    settings = _settings(exp="demo", root="deadbeef")
    identity = resolve_experiment_identity("demo")
    meta = FakeMeta(
        schema_version=INSTANCE_SCHEMA_VERSION,
        experiment_id_raw=identity.raw,
        experiment_uuid=uuid.uuid4(),
        root_commit_hash="deadbeef",
    )
    session = FakeSession(meta=meta)
    with pytest.raises(InstanceMetadataError, match="UUID mapping does not match"):
        validate_instance_marker(
            session=session,
            settings=settings,
            schema_version=INSTANCE_SCHEMA_VERSION,
        )


def test_validate_instance_marker_rejects_root_commit_mismatch() -> None:
    settings = _settings(exp="demo", root="deadbeef")
    meta = _meta_for("demo", "cafebabe")
    session = FakeSession(meta=meta)
    with pytest.raises(InstanceMetadataError, match="MAPELITES_EXPERIMENT_ROOT_COMMIT does not match"):
        validate_instance_marker(
            session=session,
            settings=settings,
            schema_version=INSTANCE_SCHEMA_VERSION,
        )


def test_resolve_instance_namespace_from_marker_uses_stored_experiment_id() -> None:
    meta = _meta_for("demo-jan21", "deadbeef")
    session = FakeSession(meta=meta)
    namespace = resolve_instance_namespace_from_marker(
        session=session,
        schema_version=INSTANCE_SCHEMA_VERSION,
    )
    assert namespace == safe_namespace_or_none("demo-jan21")


def test_seed_instance_marker_merges_expected_fields() -> None:
    settings = _settings(exp="demo", root="deadbeef")
    session = FakeSession(meta=None)
    seed_instance_marker(
        session=session,
        settings=settings,
        schema_version=INSTANCE_SCHEMA_VERSION,
    )
    assert session.merged is not None
    merged = session.merged
    identity = resolve_experiment_identity("demo")
    assert getattr(merged, "id", None) == 1
    assert getattr(merged, "schema_version", None) == INSTANCE_SCHEMA_VERSION
    assert getattr(merged, "experiment_id_raw", None) == identity.raw
    assert str(getattr(merged, "experiment_uuid", "")) == str(identity.uuid)
    assert getattr(merged, "root_commit_hash", None) == "deadbeef"

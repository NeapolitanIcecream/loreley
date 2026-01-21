from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from loreley.naming import (
    DEFAULT_TASKS_QUEUE_PREFIX,
    DEFAULT_TASKS_REDIS_NAMESPACE_PREFIX,
    DEFAULT_WORKER_JOB_BRANCH_PREFIX,
    resolve_experiment_identity,
    resolve_experiment_uuid,
    safe_namespace_from_settings,
    safe_namespace_or_none,
    tasks_queue_name,
    tasks_redis_namespace,
    worker_job_branch_prefix,
)


def test_resolve_experiment_identity_requires_value() -> None:
    with pytest.raises(ValueError, match="EXPERIMENT_ID is required"):
        resolve_experiment_identity(None)

    with pytest.raises(ValueError, match="EXPERIMENT_ID is required"):
        resolve_experiment_identity("   ")


def test_resolve_experiment_identity_uuid_uses_hex_namespace() -> None:
    exp = uuid.UUID("12345678-1234-5678-1234-567812345678")
    identity = resolve_experiment_identity(exp)
    assert identity.uuid == exp
    assert identity.namespace == exp.hex
    assert identity.raw == str(exp)


def test_resolve_experiment_identity_slug_is_stable_and_suffixed() -> None:
    raw = "Circle Pack Jan21"
    identity = resolve_experiment_identity(raw)
    identity2 = resolve_experiment_identity(raw)
    assert identity.uuid == identity2.uuid
    assert identity.namespace == identity2.namespace
    assert identity.uuid == resolve_experiment_uuid(raw)

    expected_base = "circle-pack-jan21"
    assert identity.namespace == f"{expected_base}-{identity.uuid.hex[:8]}"


def test_resolve_experiment_identity_slug_falls_back_to_exp_when_slug_is_empty() -> None:
    raw = "!!!"
    identity = resolve_experiment_identity(raw)
    assert identity.namespace.startswith("exp-")
    assert identity.namespace.endswith(identity.uuid.hex[:8])


def test_safe_namespace_or_none_handles_unset_and_uuid() -> None:
    assert safe_namespace_or_none(None) is None
    assert safe_namespace_or_none("") is None

    exp = uuid.UUID("12345678-1234-5678-1234-567812345678")
    assert safe_namespace_or_none(exp) == exp.hex


def test_derived_names_follow_expected_prefixes_for_slug() -> None:
    raw = "circle-packing"
    identity = resolve_experiment_identity(raw)

    assert tasks_redis_namespace(raw) == f"{DEFAULT_TASKS_REDIS_NAMESPACE_PREFIX}.{identity.namespace}"
    assert tasks_queue_name(raw) == f"{DEFAULT_TASKS_QUEUE_PREFIX}.{identity.namespace}"
    assert worker_job_branch_prefix(raw) == f"{DEFAULT_WORKER_JOB_BRANCH_PREFIX}/{identity.namespace}"


def test_safe_namespace_from_settings_reads_experiment_id_attr() -> None:
    assert safe_namespace_from_settings(SimpleNamespace(experiment_id=None)) is None

    raw = "demo-jan21"
    assert safe_namespace_from_settings(SimpleNamespace(experiment_id=raw)) == safe_namespace_or_none(raw)


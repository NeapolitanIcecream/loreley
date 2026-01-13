from __future__ import annotations

import uuid

from loreley.tasks.queues import DEFAULT_TASKS_QUEUE_PREFIX, experiment_queue_name


def test_experiment_queue_name_uses_prefix_and_uuid_hex() -> None:
    exp = uuid.UUID("12345678-1234-5678-1234-567812345678")
    queue = experiment_queue_name(base_queue="loreley.evolution", experiment_id=exp)
    assert queue == f"loreley.evolution.{exp.hex}"


def test_experiment_queue_name_defaults_prefix_when_missing() -> None:
    exp = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    queue = experiment_queue_name(base_queue=None, experiment_id=exp)
    assert queue == f"{DEFAULT_TASKS_QUEUE_PREFIX}.{exp.hex}"

